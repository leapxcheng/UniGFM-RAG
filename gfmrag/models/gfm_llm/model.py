from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa:N812
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model  # type: ignore[import-untyped]
from torch import nn
from torch_geometric.data import Data  # type: ignore[import-untyped]
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.utils import cached_file

from gfmrag.models.base_model import BaseGNNModel
from gfmrag.models.gfm_reasoner import GraphReasoner

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Special token for graph embeddings
GRAPH_TOKEN = "<graph_token>"
USE_GRAPH_TOKEN = True
USE_PADDING = True
USE_NODE_SAMPLER = True


class GumbelBernoulliSampler(nn.Module):
    """Differentiable Bernoulli sampling using Gumbel-Softmax relaxation with STE.

    This module enables gradient flow through discrete sampling by:
    1. Applying Gumbel-Softmax relaxation for continuous approximation
    2. Using Straight-Through Estimator (STE) for hard samples with soft gradients

    Args:
        temperature: Initial temperature for Gumbel-Softmax (default: 1.0)
        hard: Whether to use hard samples with STE (default: True)
    """

    def __init__(self, temperature: float = 1.0, hard: bool = True) -> None:
        super().__init__()
        self.temperature = temperature
        self.hard = hard

    def forward(
        self,
        logits: torch.Tensor,
        temperature: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample from Bernoulli distribution with Gumbel-Softmax relaxation.

        Args:
            logits: Raw scores from GNN (batch_size, num_nodes)
            temperature: Override temperature for this forward pass

        Returns:
            samples: Sampled binary mask (batch_size, num_nodes) - differentiable
            soft_probs: Soft probabilities (batch_size, num_nodes)
        """
        temp = temperature if temperature is not None else self.temperature

        # Convert to 2-class logits for Gumbel-Softmax: [not_selected, selected]
        # logits shape: (batch_size, num_nodes)
        logits_2class = torch.stack([torch.zeros_like(logits), logits], dim=-1)
        # logits_2class shape: (batch_size, num_nodes, 2)

        # Apply Gumbel-Softmax
        soft_samples = F.gumbel_softmax(logits_2class, tau=temp, hard=False, dim=-1)
        # soft_samples shape: (batch_size, num_nodes, 2)

        # Get probability of selection (second class)
        soft_probs = soft_samples[..., 1]  # (batch_size, num_nodes)

        if self.hard:
            # Hard samples with STE: forward uses hard, backward uses soft
            hard_samples = (soft_probs > 0.5).float()
            # STE trick: hard_samples - soft_probs.detach() + soft_probs
            samples = hard_samples - soft_probs.detach() + soft_probs
        else:
            samples = soft_probs

        return samples, soft_probs

    def set_temperature(self, temperature: float) -> None:
        """Update the temperature parameter."""
        self.temperature = temperature


class GFMLLMModel(BaseGNNModel):
    """A model that combines GraphReasoner with a Language Model for graph-enhanced QA.

    This model integrates a pre-trained GraphReasoner for graph reasoning with a
    pre-trained LLM for natural language generation. It supports flexible fine-tuning
    strategies including freezing components and LoRA adaptation.

    The model can be instantiated in multiple ways:
    1. From individual components (GraphReasoner config + LLM path)
    2. From a pretrained GFM checkpoint + LLM path
    3. Using the `from_pretrained` class method for loading saved checkpoints

    Args:
        graph_reasoner: Pre-instantiated GraphReasoner or config dict
        llm_model_name_or_path: HuggingFace model name or path
        gfm_pretrained_path: Path to pretrained GFM weights
        freeze_gfm: Whether to freeze GraphReasoner
        freeze_llm: Whether to freeze LLM
        enable_lora: Whether to use LoRA
        lora_config: LoRA configuration
        projection_hidden_dim: Hidden dim for projection layer
        llm_dtype: LLM precision type
        llm_attn_implementation: Attention implementation
        feat_dim: Feature dimension (auto-detected if not provided)
    """

    DTYPE_MAP = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "auto": "auto",
    }

    def __init__(
        self,
        graph_reasoner: GraphReasoner | dict | DictConfig | None = None,
        llm_model_name_or_path: str = "",
        gfm_pretrained_path: str = "",
        freeze_gfm: bool = True,
        freeze_llm: bool = False,
        enable_lora: bool = False,
        lora_config: dict | LoraConfig | None = None,
        projection_hidden_dim: int = 2048,
        llm_dtype: str = "bfloat16",
        llm_attn_implementation: str = "flash_attention_2",
        enable_thinking: bool = False,
        feat_dim: int | None = None,
        max_graph_tokens: int = 32,
        gumbel_temperature: float = 1.0,
        gumbel_hard: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the GFM-LLM model.

        Args:
            graph_reasoner: Pre-instantiated GraphReasoner or config dict
            llm_model_name_or_path: HuggingFace model name or path
            gfm_pretrained_path: Path to pretrained GFM weights
            freeze_gfm: Whether to freeze GraphReasoner
            freeze_llm: Whether to freeze LLM
            enable_lora: Whether to use LoRA
            lora_config: LoRA configuration
            projection_hidden_dim: Hidden dim for projection layer
            llm_dtype: LLM precision type
            llm_attn_implementation: Attention implementation
            feat_dim: Feature dimension (auto-detected if not provided)
            max_graph_tokens: Maximum number of graph tokens per sample
            gumbel_temperature: Initial temperature for Gumbel-Softmax sampling
            gumbel_hard: Whether to use hard samples with STE
        """
        super().__init__()

        self.freeze_gfm = freeze_gfm
        self.freeze_llm = freeze_llm
        self.enable_lora = enable_lora
        self.projection_hidden_dim = projection_hidden_dim
        self.llm_dtype = llm_dtype
        self.llm_attn_implementation = llm_attn_implementation
        self.max_graph_tokens = max_graph_tokens
        self.enable_thinking = enable_thinking

        # Store config for serialization
        self._init_config = {
            "llm_model_name_or_path": llm_model_name_or_path,
            "gfm_pretrained_path": gfm_pretrained_path,
            "freeze_gfm": freeze_gfm,
            "freeze_llm": freeze_llm,
            "enable_lora": enable_lora,
            "lora_config": lora_config if isinstance(lora_config, dict) else None,
            "projection_hidden_dim": projection_hidden_dim,
            "llm_dtype": llm_dtype,
            "llm_attn_implementation": llm_attn_implementation,
            "feat_dim": feat_dim,
            "max_graph_tokens": max_graph_tokens,
            "gumbel_temperature": gumbel_temperature,
            "gumbel_hard": gumbel_hard,
            **kwargs
        }

        # 1. Initialize GraphReasoner
        self.graph_reasoner = self._init_graph_reasoner(
            graph_reasoner, gfm_pretrained_path, feat_dim
        )
        self.feat_dim = self.graph_reasoner.feat_dim

        # 2. Initialize LLM
        self.llm, self.tokenizer = self._init_llm(
            llm_model_name_or_path, llm_dtype, llm_attn_implementation
        )

        # 3. Apply LoRA if enabled
        if enable_lora:
            self.llm = self._apply_lora(self.llm, lora_config)

        # 4. Initialize projection layer (graph features -> LLM embedding space)
        # Projects individual node embeddings to LLM hidden size
        llm_hidden_size = self.llm.config.hidden_size
        gfm_hidden_size = self.graph_reasoner.entity_model.dims[-1]

        if USE_GRAPH_TOKEN:
            self.graph_projection = nn.Sequential(
                nn.Linear(gfm_hidden_size * 2, projection_hidden_dim),
                nn.GELU(),
                nn.Linear(projection_hidden_dim, llm_hidden_size),
                nn.LayerNorm(llm_hidden_size),
            )

        # if USE_NODE_SAMPLER:
        #     # 5. Initialize Gumbel-Bernoulli sampler for differentiable node selection
        #     self.node_sampler = GumbelBernoulliSampler(
        #         temperature=gumbel_temperature,
        #         hard=gumbel_hard,
        #     )

        if USE_GRAPH_TOKEN:
            # 6. Store graph token id after tokenizer initialization
            self.graph_token_id = self.tokenizer.convert_tokens_to_ids(GRAPH_TOKEN)

        # 7. Apply freezing strategies
        self._apply_freezing()

        logger.info("GFMLLMModel initialized with:")
        logger.info(f"  - GraphReasoner feat_dim: {self.feat_dim}")
        logger.info(f"  - LLM: {llm_model_name_or_path}")
        logger.info(f"  - Freeze GFM: {freeze_gfm}")
        logger.info(f"  - Freeze LLM: {freeze_llm}")
        logger.info(f"  - LoRA enabled: {enable_lora}")
        logger.info(f"  - Max graph tokens: {max_graph_tokens}")
        if USE_GRAPH_TOKEN:
            logger.info(f"  - Graph token ID: {self.graph_token_id}")

    def _init_graph_reasoner(
        self,
        graph_reasoner: GraphReasoner | dict | DictConfig | None,
        gfm_pretrained_path: str,
        feat_dim: int | None,
    ) -> GraphReasoner:
        """Initialize or load the GraphReasoner component.

        Priority:
        1. If graph_reasoner is already a GraphReasoner instance, use it directly
        2. If gfm_pretrained_path is provided, load from pretrained
        3. If graph_reasoner is a config dict, instantiate via Hydra
        """
        if isinstance(graph_reasoner, GraphReasoner):
            return graph_reasoner

        if gfm_pretrained_path:
            # Load from pretrained GFM checkpoint
            logger.info(f"Loading GraphReasoner from: {gfm_pretrained_path}")
            return self._load_gfm_pretrained(gfm_pretrained_path)

        if graph_reasoner is not None:
            # Instantiate from config
            logger.info("Instantiating GraphReasoner from config")
            if isinstance(graph_reasoner, (dict, DictConfig)):
                if feat_dim is not None:
                    graph_reasoner = dict(graph_reasoner)  # Make mutable copy
                    graph_reasoner["feat_dim"] = feat_dim
                return instantiate(graph_reasoner)
            raise TypeError(f"Unsupported graph_reasoner type: {type(graph_reasoner)}")

        raise ValueError(
            "Must provide either graph_reasoner (instance or config) or gfm_pretrained_path"
        )

    def _load_gfm_pretrained(self, path: str) -> GraphReasoner:
        """Load GraphReasoner from a pretrained checkpoint."""
        config_path = cached_file(path, "config.json")
        if config_path is None:
            raise FileNotFoundError(f"config.json not found in {path}")

        with open(config_path) as f:
            config = json.load(f)

        model = instantiate(config["model_config"])

        model_path = cached_file(path, "model.pth")
        if model_path is None:
            raise FileNotFoundError(f"model.pth not found in {path}")

        state = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])

        return model

    def _init_llm(
        self,
        model_name_or_path: str,
        dtype: str,
        attn_implementation: str,
    ) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Initialize the LLM and tokenizer with graph token support."""
        torch_dtype = self.DTYPE_MAP.get(dtype, torch.bfloat16)

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            # padding_side="left",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if USE_GRAPH_TOKEN:
            # Add graph token as a special token
            num_added = tokenizer.add_special_tokens(
                {"additional_special_tokens": [GRAPH_TOKEN]}
            )
            logger.info(f"Added {num_added} special token(s): {GRAPH_TOKEN}")

        llm = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        )

        # Resize token embeddings to accommodate the new special token
        if USE_GRAPH_TOKEN and num_added > 0:
            llm.resize_token_embeddings(len(tokenizer))
            logger.info(f"Resized token embeddings to {len(tokenizer)}")

        return llm, tokenizer

    def _apply_lora(
        self,
        model: AutoModelForCausalLM,
        lora_config: dict | LoraConfig | None,
    ) -> AutoModelForCausalLM:
        """Apply LoRA adaptation to the LLM."""
        if lora_config is None:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
        else:
            lora_config_dict = OmegaConf.to_container(lora_config, resolve=True)
            lora_config = LoraConfig(**lora_config_dict, task_type="CAUSAL_LM")

        return get_peft_model(model, lora_config)

    def _apply_freezing(self) -> None:
        """Apply freezing strategies to model components."""
        if self.freeze_gfm:
            for param in self.graph_reasoner.parameters():
                param.requires_grad = False
            logger.info("GraphReasoner parameters frozen")

        if self.freeze_llm and not self.enable_lora:
            for param in self.llm.parameters():
                param.requires_grad = False
            logger.info("LLM parameters frozen")

    def build_prompt_with_graph_tokens(
        self,
        question: str,
        node_texts: list[str],
        system_prompt: str | None = None,
        answer: str | None = None,
    ) -> dict[str, Any]:
        """Build a chat prompt with graph tokens using the tokenizer's chat template.

        Each sampled node gets its own <graph_token> followed by its text content.
        The format in the user message will be:
            <graph_token> node_text_1
            <graph_token> node_text_2
            ...

            Question: {question}

        Args:
            question: The question to ask
            node_texts: List of node text contents for sampled nodes
            system_prompt: Optional system prompt
            answer: Optional answer for training (creates assistant turn)

        Returns:
            Dict with input_ids, attention_mask, labels (if answer provided),
            and graph_token_positions
        """
        # Build context with graph tokens
        context_parts = []
        for node_text in node_texts:
            if USE_GRAPH_TOKEN:
                context_parts.append(f"{GRAPH_TOKEN} {node_text}")
            else:
                context_parts.append(f"{node_text}")
        context_str = "\n".join(context_parts)

        # Build user message content
        user_content = f"Context:\n{context_str}\n\nQuestion: {question}"   #  /no_think

        # Build message list for chat template
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        if answer is not None:
            messages.append({"role": "assistant", "content": answer})
            # Tokenize with labels for training
            prompt_with_answer = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=False,
                enable_thinking=self.enable_thinking,
            )
            # Create labels: -100 for non-answer tokens
            # Get the prompt without answer to find where answer starts
            messages_no_answer = messages[:-1]
            prompt_only = self.tokenizer.apply_chat_template(
                messages_no_answer,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            prompt_len = len(prompt_only)

            input_ids = prompt_with_answer["input_ids"]
            attention_mask = prompt_with_answer["attention_mask"]

            # Labels: -100 for prompt, actual ids for answer
            labels = input_ids.clone()
            labels[0, :prompt_len] = -100
            # self.tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=False)
            # outputs: <think>\n\n</think>\n\n1920<|im_end|>\n
        else:
            # Tokenize for inference (no labels)
            prompt_only = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            input_ids = prompt_only["input_ids"]
            attention_mask = prompt_only["attention_mask"]
            labels = None

        if USE_GRAPH_TOKEN:
            # Find graph token positions in input_ids
            graph_token_positions = (input_ids == self.graph_token_id).nonzero(
                as_tuple=False
            )
            result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "graph_token_positions": graph_token_positions,
                "num_graph_tokens": len(node_texts),
            }
        
        else:
            result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

        if labels is not None:
            result["labels"] = labels

        return result

    def _select_top_k_nodes(
        self,
        sample_weights: torch.Tensor,
        gnn_latent: torch.Tensor,
        node_texts: list[str],
        k: int,
    ) -> tuple[list[int], torch.Tensor, list[str]]:
        """Select top-k nodes based on sampling weights.

        Args:
            sample_weights: Sampling weights (num_nodes,) - from Gumbel-Softmax
            gnn_latent: Node embeddings (num_nodes, gnn_dim)
            node_texts: List of node text strings
            k: Maximum number of nodes to select

        Returns:
            Tuple of (selected_indices, selected_embeddings, selected_texts)
        """
        # Get top-k indices by sampling weight
        num_nodes = sample_weights.size(0)
        actual_k = min(k, num_nodes)

        top_k_weights, top_k_indices = sample_weights.topk(actual_k)

        # w threshold
        # Filter to nodes with non-negligible weight (> 0.01)
        # mask = top_k_weights > 0.01
        # selected_indices_tensor = top_k_indices[mask]
        # selected_indices: list[int] = selected_indices_tensor.tolist()

        # just top-k document
        selected_indices_tensor = top_k_indices
        selected_indices: list[int] = selected_indices_tensor.tolist()

        if len(selected_indices) == 0:
            # Fallback: take at least top-1
            selected_indices = [int(top_k_indices[0].item())]

        # Gather embeddings and texts
        selected_embeddings = gnn_latent[
            selected_indices_tensor
            if len(selected_indices_tensor) > 0
            else top_k_indices[:1]
        ]
        selected_texts = [node_texts[i] for i in selected_indices]

        return selected_indices, selected_embeddings, selected_texts

    def _replace_graph_tokens_with_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        graph_embeddings: torch.Tensor,
        graph_token_positions: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Replace graph token positions with projected graph embeddings.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            graph_embeddings: Projected node embeddings (total_graph_tokens, llm_hidden)
            graph_token_positions: Positions of graph tokens (num_positions, 2) where
                                   each row is [batch_idx, seq_idx]
            labels: Optional labels (batch_size, seq_len)

        Returns:
            Tuple of (inputs_embeds, attention_mask, labels)
        """
        # Get base embeddings from LLM
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        # inputs_embeds shape: (batch_size, seq_len, llm_hidden)

        # Replace graph token embeddings with projected graph embeddings
        # graph_token_positions has shape (num_graph_tokens, 2)
        # graph_embeddings has shape (num_graph_tokens, llm_hidden)
        for i, (batch_idx, seq_idx) in enumerate(graph_token_positions):
            inputs_embeds[batch_idx, seq_idx] = graph_embeddings[i]

        # Labels for graph token positions should be -100 (not predicted)
        if labels is not None:
            for batch_idx, seq_idx in graph_token_positions:
                labels[batch_idx, seq_idx] = -100

        return inputs_embeds, attention_mask, labels

    def add_gumbel_noise(self, logits, dim=-1, eps=1e-20):
        
        u = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(u + eps) + eps)    # 
        soft_probs = logits + gumbel_noise

        return soft_probs

    def forward(
        self,
        graph: Data,
        batch: dict[str, torch.Tensor],
        questions: list[str] | None = None,
        answers: list[str] | None = None,
        system_prompt: str | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        num_graph_tokens_per_sample: list[int] | None = None,
        graph_token_positions: torch.Tensor | None = None,
        temperature: float | None = None,
        node_texts: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the GFM-LLM model with differentiable node sampling.

        This method supports two modes:
        1. End-to-end mode: Provide questions/answers, model builds prompts internally
        2. Pre-tokenized mode: Provide input_ids/attention_mask/labels directly

        Args:
            graph: PyG Data object containing the graph structure
            batch: Dict containing question_embeddings, start_nodes_mask, etc.
            questions: List of question strings (for end-to-end mode)
            answers: List of answer strings (for training in end-to-end mode)
            system_prompt: Optional system prompt
            input_ids: Pre-tokenized input (for pre-tokenized mode)
            attention_mask: Attention mask (for pre-tokenized mode)
            labels: Labels (for pre-tokenized mode)
            num_graph_tokens_per_sample: Number of graph tokens per sample (pre-tokenized)
            graph_token_positions: Positions of graph tokens (pre-tokenized)
            temperature: Override Gumbel-Softmax temperature
            node_texts: List of node text strings for prompt building

        Returns:
            Dict containing loss, logits, graph_scores, sample_weights
        """
        batch_size = batch["question_embeddings"].size(0)
        device = batch["question_embeddings"].device

        # 1. Get graph representations from GraphReasoner
        graph_scores, gnn_latent = self.graph_reasoner(
            graph, batch, return_gnn_latent=True
        )
        # graph_scores: (batch_size, num_nodes)
        # gnn_latent: (batch_size, num_nodes, gnn_dim)
        
        # just keep document node
        target_node_ids = graph.nodes_by_type['document']
        graph_scores = graph_scores[:, target_node_ids]
        gnn_latent = gnn_latent[:, target_node_ids]
        node_texts = [node_texts[i] for i in target_node_ids] # node_texts[:len(target_node_ids)]

        question_embeddings = batch['question_embeddings']
        doc_embeddings = graph.x[target_node_ids]

        q_norm = F.normalize(question_embeddings, p=2, dim=-1)
        n_norm = F.normalize(doc_embeddings, p=2, dim=-1)
        teacher_scores = torch.matmul(q_norm, n_norm.transpose(0, 1)) # [batch_size, num_target_nodes]

        loss_type = self._init_config.get("loss_type", None)
        # alpha = self._init_config.get("alpha", 1.0)
        alpha = self._init_config["alpha"]
        distill_tau = self._init_config.get("distill_temperature", 0.05)
        if loss_type == 'kl':
                        
            if self._init_config.get("loss_if_top_k"):
                teacher_topk_scores, topk_indices = torch.topk(teacher_scores, k=self._init_config.get("loss_top_k"), dim=-1)
                student_topk_scores = torch.gather(graph_scores, dim=1, index=topk_indices)
                
                teacher_probs = F.softmax(teacher_topk_scores / distill_tau, dim=-1)
                student_log_probs = F.log_softmax(student_topk_scores / distill_tau, dim=-1)
                
                gnn_loss = F.kl_div(
                    student_log_probs, 
                    teacher_probs, 
                    reduction='batchmean'
                )

            else:
                teacher_probs = F.softmax(teacher_scores / distill_tau, dim=-1)
                student_log_probs = F.log_softmax(graph_scores / 1.0, dim=-1)
                
                gnn_loss = F.kl_div(
                    student_log_probs, 
                    teacher_probs, 
                    reduction='batchmean'
                )

                # # g-reasoner distill loss
                # ## sigmoid
                # student_log_probs = F.sigmoid(graph_scores)
                # teacher_probs = (teacher_scores + 1) / 2
                
                # student_prob = torch.clamp(student_log_probs, min=1e-6, max=1 - 1e-6)
                # target_prob = torch.clamp(teacher_probs, min=1e-6, max=1 - 1e-6)
                # # Compute the KL divergence loss
                # loss = target_prob * (torch.log(target_prob) - torch.log(student_prob)) + (
                #     1 - target_prob
                # ) * (torch.log(1 - target_prob) - torch.log(1 - student_prob))

                # gnn_loss = loss.sum() / graph_scores.size(0)
        
        elif loss_type == 'rkl':
            teacher_log_probs = F.log_softmax(teacher_scores / distill_tau, dim=-1)
            student_probs = F.softmax(graph_scores / 1.0, dim=-1)
            
            gnn_loss = F.kl_div(
                teacher_log_probs,
                student_probs, 
                reduction='batchmean'
            )       # reverse kl

        elif loss_type == 'mse':
            gnn_loss = F.mse_loss(graph_scores, teacher_scores)

        if USE_NODE_SAMPLER:
            # w gumbel-softmax
            # 2. Sample nodes using Gumbel-Softmax + STE

            # directly use sigmod functions
            # sample_weights = graph_scores

            # sample_weights = self.add_gumbel_noise(graph_scores)      # add gumbel noise
            sample_weights = graph_scores
            soft_probs = torch.sigmoid(sample_weights)

            # sample_weights, soft_probs = self.node_sampler(graph_scores, temperature)
            # sample_weights: (batch_size, num_nodes) - differentiable binary-like mask
        else:
            # w/o gumbel-softmax
            sample_weights = graph_scores
            soft_probs = graph_scores

        outputs = {
            "graph_scores": graph_scores,
            "sample_weights": sample_weights,
            "soft_probs": soft_probs,
        }

        # 3. Handle two modes: end-to-end or pre-tokenized
        if questions is not None:
            # End-to-end mode: build prompts from questions and sampled nodes
            assert node_texts is not None, (
                "node_texts must be provided for end-to-end mode"
            )

            all_input_ids = []
            all_attention_masks = []
            all_labels = []
            all_graph_token_positions = []
            all_selected_embeddings = []

            for i in range(batch_size):
                # Select top-k nodes for this sample based on sampling weights
                selected_indices, selected_embeddings, selected_texts = (
                    self._select_top_k_nodes(
                        sample_weights[i],
                        gnn_latent[i],
                        node_texts,
                        self.max_graph_tokens,
                    )
                )

                if USE_NODE_SAMPLER:    
                    # Weight embeddings by their soft probabilities for gradient flow
                    # top-k softmax weights
                    weight_type = self._init_config.get("weight_type", None)
                    if weight_type == 'top_k_softmax':
                        soft_weights = graph_scores[i, selected_indices].unsqueeze(-1)
                        soft_weights = torch.softmax(soft_weights, dim=-1)
                    elif weight_type == 'sigmoid':

                        soft_weights = soft_probs[i, selected_indices].unsqueeze(-1)
                    weighted_embeddings = selected_embeddings * soft_weights
                else:
                    weighted_embeddings = selected_embeddings

                # Build prompt with graph tokens
                answer = answers[i] if answers is not None else None
                prompt_result = self.build_prompt_with_graph_tokens(
                    question=questions[i],
                    node_texts=selected_texts,
                    system_prompt=system_prompt,
                    answer=answer,
                )

                all_input_ids.append(prompt_result["input_ids"])
                all_attention_masks.append(prompt_result["attention_mask"])
                if "labels" in prompt_result:
                    all_labels.append(prompt_result["labels"])

                if USE_GRAPH_TOKEN:
                    # Adjust graph token positions for batch indexing
                    positions = prompt_result["graph_token_positions"].clone()
                    positions[:, 0] = i  # Set batch index
                    all_graph_token_positions.append(positions)
                    all_selected_embeddings.append(weighted_embeddings)
            
            if USE_PADDING:
                # Pad sequences to same length
                input_ids = self._pad_sequences(all_input_ids, self.tokenizer.pad_token_id, mode='right')
                attention_mask = self._pad_sequences(all_attention_masks, 0, mode='right')
                if all_labels:
                    labels = self._pad_sequences(all_labels, -100, mode='right')
                else:
                    labels = None
            else:
                input_ids = all_input_ids[0]
                attention_mask = all_attention_masks[0]
                if all_labels:
                    labels = all_labels[0]

            if USE_GRAPH_TOKEN:
                # Concatenate graph token positions and embeddings
                graph_token_positions = torch.cat(all_graph_token_positions, dim=0)
                graph_embeddings = torch.cat(all_selected_embeddings, dim=0)
                graph_token_positions = graph_token_positions.to(device)

            # Move to device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            if labels is not None:
                labels = labels.to(device)

        else:
            # Pre-tokenized mode: use provided inputs
            assert input_ids is not None, "Must provide either questions or input_ids"
            assert graph_token_positions is not None, (
                "Must provide graph_token_positions for pre-tokenized mode"
            )
            assert num_graph_tokens_per_sample is not None, (
                "Must provide num_graph_tokens_per_sample for pre-tokenized mode"
            )

            # Gather embeddings for each graph token position
            all_embeddings = []
            token_idx = 0
            for i, num_tokens in enumerate(num_graph_tokens_per_sample):
                # Get top-k node indices for this sample
                _, top_k_indices = sample_weights[i].topk(num_tokens)
                selected_embeddings = gnn_latent[i, top_k_indices]
                soft_weights = soft_probs[i, top_k_indices].unsqueeze(-1)
                weighted_embeddings = selected_embeddings * soft_weights
                all_embeddings.append(weighted_embeddings)
                token_idx += num_tokens

            graph_embeddings = torch.cat(all_embeddings, dim=0)

        if USE_GRAPH_TOKEN:
            # 4. Project graph embeddings to LLM space
            projected_embeddings = self.graph_projection(graph_embeddings)
            # projected_embeddings: (total_graph_tokens, llm_hidden)

        if USE_GRAPH_TOKEN:
            # 5. Replace graph tokens with projected embeddings
            inputs_embeds, attention_mask, labels = (
                self._replace_graph_tokens_with_embeddings(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    graph_embeddings=projected_embeddings,
                    graph_token_positions=graph_token_positions,
                    labels=labels,
                )
            )

            # 6. LLM forward pass
            llm_outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
        
        else:
            llm_outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
                )

        outputs['llm_loss'] = llm_outputs.loss
        outputs['gnn_loss'] = gnn_loss
        outputs["loss"] = llm_outputs.loss + alpha * gnn_loss
        # outputs["loss"] = alpha * llm_outputs.loss + gnn_loss
        outputs["logits"] = llm_outputs.logits

        return outputs

    def _pad_sequences(
        self,
        sequences: list[torch.Tensor],
        pad_value: int,
        mode: str
    ) -> torch.Tensor:
        """Pad a list of 2D tensors to the same length (left padding for causal LM).

        Args:
            sequences: List of tensors, each (1, seq_len_i)
            pad_value: Value to use for padding

        Returns:
            Padded tensor (batch_size, max_seq_len)
        """
        max_len = max(seq.size(1) for seq in sequences)
        padded = []
        for seq in sequences:
            pad_len = max_len - seq.size(1)
            if pad_len > 0:
                padding = torch.full(
                    (1, pad_len), pad_value, dtype=seq.dtype, device=seq.device
                )
                if mode == 'left':
                    seq = torch.cat([padding, seq], dim=1)
                elif mode == 'right':
                    seq = torch.cat([seq, padding], dim=1)
            padded.append(seq)
        return torch.cat(padded, dim=0)

    @torch.inference_mode()
    def generate(
        self,
        graph: Data,
        batch: dict[str, torch.Tensor],
        questions: list[str] | None = None,
        system_prompt: str | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        num_graph_tokens_per_sample: list[int] | None = None,
        graph_token_positions: torch.Tensor | None = None,
        max_new_tokens: int = 256,
        threshold: float = 0.5,
        top_k: int | None = None,
        max_graph_tokens: int = 32,
        node_texts: list[str] | None = None,
        return_retrieved_nodes: bool = False,
        **generate_kwargs: Any,
    ) -> list[torch.Tensor, list[list[int]] | None]:
        """Generate text conditioned on graph context.

        Args:
            graph: PyG Data object
            batch: Dict with question embeddings and masks
            questions: List of question strings (for end-to-end mode)
            system_prompt: Optional system prompt
            input_ids: Pre-tokenized prompt (for pre-tokenized mode)
            attention_mask: Attention mask (for pre-tokenized mode)
            num_graph_tokens_per_sample: Number of graph tokens per sample
            graph_token_positions: Positions of graph tokens
            max_new_tokens: Maximum tokens to generate
            threshold: Probability threshold for node selection
            max_graph_tokens: Maximum number of graph tokens to include
            top_k: Top-K nodes to consider during generation
            node_texts: List of node text strings for prompt building
            **generate_kwargs: Additional generation arguments

        Returns:
            Generated token ids
            retrieved_nodes: List of retrieved node indices per sample (if requested)
        """
        batch_size = batch["question_embeddings"].size(0)
        device = batch["question_embeddings"].device

        # Get graph representations
        graph_scores, gnn_latent = self.graph_reasoner(
            graph, batch, return_gnn_latent=True
        )
        
        # just document
        target_node_ids = graph.nodes_by_type['document']
        graph_scores = graph_scores[:, target_node_ids]
        gnn_latent = gnn_latent[:, target_node_ids]

        probs = torch.sigmoid(graph_scores)

        if return_retrieved_nodes:
            retrieved_nodes = []
        else:
            retrieved_nodes = None

        if questions is not None:
            # End-to-end mode
            assert node_texts is not None, (
                "node_texts must be provided for end-to-end mode"
            )

            all_input_ids = []
            all_attention_masks = []
            all_graph_token_positions = []
            all_selected_embeddings = []

            for i in range(batch_size):
                # Select nodes above threshold or top-k
                node_probs_i = probs[i]
                node_scores_i = graph_scores[i]
                if top_k is not None:
                    _, top_k_indices = node_scores_i.topk(
                        min(top_k, node_scores_i.size(0))
                    )
                    selected_indices = top_k_indices.tolist()

                selected_embeddings = gnn_latent[i, selected_indices]
                selected_texts = [node_texts[j] for j in selected_indices]

                if USE_NODE_SAMPLER:    
                    weight_type = self._init_config.get("weight_type", None)
                    if weight_type == 'top_k_softmax':
                        soft_weights = node_scores_i[selected_indices].unsqueeze(-1)
                        soft_weights = torch.softmax(soft_weights, dim=-1)
                    elif weight_type == 'sigmoid':
                        soft_weights = node_probs_i[selected_indices].unsqueeze(-1)

                    weighted_embeddings = selected_embeddings * soft_weights

                if return_retrieved_nodes:
                    retrieved_nodes.append(selected_indices)

                # Build prompt
                prompt_result = self.build_prompt_with_graph_tokens(
                    question=questions[i],
                    node_texts=selected_texts,
                    system_prompt=system_prompt,
                    answer=None,
                )

                all_input_ids.append(prompt_result["input_ids"])
                all_attention_masks.append(prompt_result["attention_mask"])

                if USE_GRAPH_TOKEN:
                    positions = prompt_result["graph_token_positions"].clone()
                    positions[:, 0] = i
                    all_graph_token_positions.append(positions)
                    all_selected_embeddings.append(weighted_embeddings)

            if USE_PADDING:
                # Pad sequences
                input_ids = self._pad_sequences(all_input_ids, self.tokenizer.pad_token_id, mode='left')
                attention_mask = self._pad_sequences(all_attention_masks, 0, mode='left')
            else:
                input_ids = all_input_ids[0]
                attention_mask = all_attention_masks[0]

            if USE_GRAPH_TOKEN:
                graph_token_positions = torch.cat(all_graph_token_positions, dim=0)
                graph_embeddings = torch.cat(all_selected_embeddings, dim=0)
                graph_token_positions = graph_token_positions.to(device)

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
        else:
            # Pre-tokenized mode
            assert input_ids is not None
            assert graph_token_positions is not None
            assert num_graph_tokens_per_sample is not None

            all_embeddings = []
            for i, num_tokens in enumerate(num_graph_tokens_per_sample):
                _, top_k_indices = probs[i].topk(num_tokens)
                all_embeddings.append(gnn_latent[i, top_k_indices])
            graph_embeddings = torch.cat(all_embeddings, dim=0)

        if USE_GRAPH_TOKEN:
            # Project embeddings
            projected_embeddings = self.graph_projection(graph_embeddings)

            # Replace graph tokens
            inputs_embeds, attention_mask, _ = self._replace_graph_tokens_with_embeddings(
                input_ids=input_ids,
                attention_mask=attention_mask,
                graph_embeddings=projected_embeddings,
                graph_token_positions=graph_token_positions,
            )

            # Generate
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                **generate_kwargs,
            )
        
        else:
            input_length = input_ids.shape[1]
            outputs = self.llm.generate(
                input_ids=input_ids,          # <--- 注意这里变成了 input_ids
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                **generate_kwargs,
            )
            outputs = outputs[:, input_length:]

        return outputs, retrieved_nodes

    def save_pretrained(self, save_path: str) -> None:
        """Save the model in a format that can be loaded with from_pretrained.

        Args:
            save_path: Directory to save the model
        """
        os.makedirs(save_path, exist_ok=True)

        # Save config
        config = {
            "model_type": "GFMLLMModel",
            "init_config": self._init_config,
            "graph_reasoner_config": OmegaConf.to_container(
                self._init_config.get("graph_reasoner_config", {})
            )
            if hasattr(self, "_graph_reasoner_config")
            else None,
        }
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        if USE_GRAPH_TOKEN:
            # Save model weights
            state_dict = {
                "graph_reasoner": self.graph_reasoner.state_dict(),
                "graph_projection": self.graph_projection.state_dict(),
            }
            torch.save(state_dict, os.path.join(save_path, "gfm_weights.pth"))
        else:
            # Save model weights
            state_dict = {
                "graph_reasoner": self.graph_reasoner.state_dict(),
            }
            torch.save(state_dict, os.path.join(save_path, "gfm_weights.pth"))

        # Save LLM (or LoRA adapter)
        if self.enable_lora:
            self.llm.save_pretrained(os.path.join(save_path, "llm_adapter"))
        else:
            self.llm.save_pretrained(os.path.join(save_path, "llm"))

        logger.info(f"Model saved to {save_path}")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        llm_model_name_or_path: str | None = None,
        **override_kwargs: Any,
    ) -> GFMLLMModel:
        """Load a pretrained GFMLLMModel.

        Args:
            pretrained_path: Path to saved model directory
            llm_model_name_or_path: Override LLM path (useful for loading base LLM)
            **override_kwargs: Override init arguments

        Returns:
            Loaded GFMLLMModel instance
        """
        # Load config
        config_path = os.path.join(pretrained_path, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        init_config = config["init_config"]

        # Override with provided arguments
        if llm_model_name_or_path:
            init_config["llm_model_name_or_path"] = llm_model_name_or_path
        init_config.update(override_kwargs)

        # Create model instance
        model = cls(**init_config)

        # Load GFM weights
        gfm_weights_path = os.path.join(pretrained_path, "gfm_weights.pth")
        if os.path.exists(gfm_weights_path):
            state_dict = torch.load(
                gfm_weights_path, map_location="cpu", weights_only=True
            )
            model.graph_reasoner.load_state_dict(state_dict["graph_reasoner"])
            if USE_GRAPH_TOKEN:
                model.graph_projection.load_state_dict(state_dict["graph_projection"])

        # Load LLM/adapter weights
        llm_adapter_path = os.path.join(pretrained_path, "llm_adapter")
        llm_path = os.path.join(pretrained_path, "llm")

        if os.path.exists(llm_adapter_path) and init_config.get("enable_lora", False):
            from peft import PeftModel

            model.llm = PeftModel.from_pretrained(model.llm, llm_adapter_path)
        elif os.path.exists(llm_path):
            model.llm = AutoModelForCausalLM.from_pretrained(llm_path)

        logger.info(f"Model loaded from {pretrained_path}")
        return model

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Get list of trainable parameters."""
        trainable = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable.append(param)
                logger.debug(f"Trainable: {name}")
        return trainable

    def print_trainable_parameters(self) -> None:
        """Print statistics about trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
