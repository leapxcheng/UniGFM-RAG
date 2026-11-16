"""Trainer for end-to-end GFM-LLM model with differentiable node sampling."""

import logging
import os
from dataclasses import dataclass
from itertools import islice
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from gfmrag import utils
from gfmrag.evaluation.qa_metrics import (
    QAMetric,
    get_qa_metric,
)
from gfmrag.graph_index_datasets.graph_dataset_loader import (
    GraphDataset,
    GraphDatasetLoader,
)
from gfmrag.models.ultra import query_utils
from gfmrag.utils.wandb_utils import log_model_checkpoint

from .base_trainer import BaseTrainer, TaskDataset
from .training_args import TrainingArguments

logger = logging.getLogger(__name__)

@dataclass
class GFMLLMTrainingArguments(TrainingArguments):
    """Training arguments specific to GFM-LLM model."""
    
    gradient_accumulation_steps: int = 1

    max_grad_norm: float | None = None  # Gradient clipping norm

    # Gumbel-Softmax temperature scheduling
    initial_temperature: float = 1.0
    final_temperature: float = 0.1
    temperature_anneal_epochs: int | None = None  # If None, anneal over all epochs

    # System prompt for QA
    system_prompt: str = "You are a helpful assistant that answers questions based on the provided context. Please only respond with the answer and no additional text."

    # Auxiliary loss weight for GNN supervision (optional)
    gnn_loss_weight: float = 0.0  # Set > 0 to add supervision on GNN scores

    max_new_tokens: int = 128  # Max new tokens to generate during evaluation
    max_graph_tokens: int = 5  # Max graph tokens to include during generation
    threshold: float = 0.5  # Probability threshold for node selection during generation
    top_k: int | None = (
        None  # Top-K nodes to consider during generation # Once set the threshold and max_graph_tokens will be ignored and only top_k nodes will be selected
    )


@dataclass
class GraphLLMTaskDataset(TaskDataset):
    """Task-specific dataset for GFM-RAG models."""

    node_texts: list[str] | None = None  # Optional: node text for LLM prompt building
    raw_data: dict[str, Any] | None = (
        None  # Optional: raw QA data for question/answer lookup
    )


class GFMLLMTrainer(BaseTrainer):
    """Trainer for end-to-end GFM-LLM model.

    This trainer handles:
    1. End-to-end training with LLM loss backpropagating through differentiable
       Gumbel-Softmax sampling to the GNN
    2. Temperature annealing for Gumbel-Softmax during training
    3. Optional auxiliary GNN supervision loss
    4. QA evaluation with configurable metrics (EM, FM, F1)
    """

    def __init__(
        self,
        output_dir: str,
        args: GFMLLMTrainingArguments,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_graph_dataset_loader: GraphDatasetLoader | None = None,
        eval_graph_dataset_loader: GraphDatasetLoader | None = None,
        gnn_loss_fn: nn.Module | None = None,
        qa_metrics: list[str] | None = None,
        re_metrics: list[str] | None = None,
        prompt_map: dict[str, str] | None=None,
        **kwargs: Any,
    ) -> None:
        """Initialize the GFM-LLM trainer.

        Args:
            output_dir: Directory to save checkpoints
            args: Training arguments
            model: GFMLLMModel instance
            optimizer: Optimizer for training
            train_graph_dataset_loader: Loader for training data
            eval_graph_dataset_loader: Loader for evaluation data
            gnn_loss_fn: Optional loss function for GNN supervision
            qa_metrics: List of QA metric names to compute (e.g., ["em", "fm", "f1"])
            re_metrics: List of retrieval metric names to compute (e.g., ["Recall@2", "Recall@5", "Recall@10"])
        """
        # Set default metric for best model
        if args.metric_for_best_model is None:
            args.metric_for_best_model = "eval_loss"

        super().__init__(
            output_dir=output_dir,
            model=model,
            args=args,
            train_graph_dataset_loader=train_graph_dataset_loader,
            eval_graph_dataset_loader=eval_graph_dataset_loader,
            optimizer=optimizer,
            **kwargs,
        )

        self.gnn_loss_fn = gnn_loss_fn
        self.gnn_loss_weight = args.gnn_loss_weight
        self.system_prompt = args.system_prompt

        # Initialize QA metrics
        if qa_metrics is None:
            qa_metrics = ["em", "fm", "f1"]  # Default metrics
        self.qa_metrics: dict[str, QAMetric] = {
            name: get_qa_metric(name) for name in qa_metrics
        }
        self.re_metrics = re_metrics
        self.prompt_map = prompt_map

        # Temperature scheduling
        self.initial_temperature = args.initial_temperature
        self.final_temperature = args.final_temperature
        self.temperature_anneal_epochs = (
            args.temperature_anneal_epochs or args.num_epoch
        )

        self.lr_scheduler = kwargs['lr_scheduler']

        logger.info("GFMLLMTrainer initialized:")
        logger.info(f"  - Initial temperature: {self.initial_temperature}")
        logger.info(f"  - Final temperature: {self.final_temperature}")
        logger.info(f"  - Temperature anneal epochs: {self.temperature_anneal_epochs}")
        logger.info(f"  - GNN loss weight: {self.gnn_loss_weight}")
        logger.info(f"  - QA metrics: {list(self.qa_metrics.keys())}")
        logger.info(f"  - Retrieval metrics: {list(self.re_metrics)}")

    def _get_current_temperature(self) -> float:
        """Calculate current Gumbel-Softmax temperature based on epoch."""
        epoch = self.state["epoch"]
        if epoch >= self.temperature_anneal_epochs:
            return self.final_temperature

        # Linear annealing
        progress = epoch / self.temperature_anneal_epochs
        temperature = self.initial_temperature - progress * (
            self.initial_temperature - self.final_temperature
        )
        return temperature

    def _create_task_dataset(
        self, dataset: GraphDataset, is_train: bool = True
    ) -> GraphLLMTaskDataset:
        """Create a task dataset for GFM-LLM training.

        The dataset should contain:
        - question_embeddings: Embeddings of questions
        - start_nodes_mask: Mask for starting nodes in the graph
        - target_nodes_mask: Mask for target nodes (optional, for GNN supervision)
        - questions: Raw question strings
        - answers: Raw answer strings
        """
        data_name = dataset.name
        sft_dataset = dataset.data
        # print(data_name)
        data = sft_dataset.train_data if is_train else sft_dataset.test_data

        sampler = torch.utils.data.DistributedSampler(
            data,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=is_train,
        )

        batch_size = (
            self.args.train_batch_size if is_train else self.args.eval_batch_size
        )
        data_loader = DataLoader(
            data,
            batch_size=batch_size,
            sampler=sampler,
        )

        # Load raw data for question/answer lookup
        raw_data = sft_dataset.raw_train_data if is_train else sft_dataset.raw_test_data

        # Load node texts from nodes.csv for LLM prompt building
        node_texts = (
            sft_dataset.raw_node_texts
            if hasattr(sft_dataset, "raw_node_texts")
            else None
        )

        return GraphLLMTaskDataset(
            name=data_name,
            graph=sft_dataset.graph.to(self.device),
            data_loader=data_loader,
            node_texts=node_texts,
            raw_data=raw_data,
        )

    def _get_questions_and_answers(
        self,
        batch: dict[str, torch.Tensor],
        raw_data: dict[str, Any],
    ) -> tuple[list[str], list[str]]:
        """Extract questions and answers from batch using raw data.

        Args:
            batch: Batch containing sample IDs
            raw_data: Raw data dict with questions and answers

        Returns:
            Tuple of (questions, answers) lists
        """
        sample_ids = batch["id"]
        questions = [raw_data[str(sid)]["question"] for sid in sample_ids]
        answers = [raw_data[str(sid)]["answer"] for sid in sample_ids]
        return questions, answers

    def train_step(
        self, batch: Any, task_dataset: GraphLLMTaskDataset # type: ignore[override]
    ) -> dict[str, float | torch.Tensor]: 
        """Perform a single training step.

        Args:
            batch: Training batch containing embeddings and masks
            task_dataset: Task dataset with graph, node_texts and raw data

        Returns:
            Dictionary with loss and metrics
        """
        graph = task_dataset.graph.to(self.device)
        batch = query_utils.cuda(batch, device=self.device)

        # Get current temperature for Gumbel-Softmax
        temperature = self._get_current_temperature()

        # Get questions and answers from batch
        # Note: This assumes the batch contains 'question' and 'answer' fields
        # or we need to look them up from raw_data using IDs
        questions, answers = self._get_questions_and_answers(
            batch, task_dataset.raw_data
        )

        # Forward pass through GFM-LLM model
        outputs = self.parallel_model(
            graph=graph,
            batch=batch,
            questions=questions,
            answers=answers,
            system_prompt=self.system_prompt,
            temperature=temperature,
            node_texts=task_dataset.node_texts,
        )

        # Primary loss: LLM language modeling loss
        total_loss = outputs["loss"]
        step_metrics = {
            "llm_loss": outputs["llm_loss"],
            "gnnn_loss": outputs["gnn_loss"],
            "lm_loss": outputs["llm_loss"].item(),
            "gnn_loss": outputs["gnn_loss"].item()
        }
        

        step_metrics["loss"] = total_loss
        step_metrics["temperature"] = temperature

        # Track sampling statistics
        sample_weights = outputs["sample_weights"]
        step_metrics["avg_selected_nodes"] = (
            (sample_weights > self.args.threshold).float().sum(dim=-1).mean().item()
        )
        step_metrics["avg_soft_prob"] = outputs["soft_probs"].mean().item()

        return step_metrics

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Evaluate the model on validation datasets.

        Uses model.generate() to get retrieved nodes and generated answers,
        then computes retrieval and QA metrics based on the generation output.

        Default metrics:
        - avg_selected_nodes: Average number of nodes selected during generation
        - gnn_recall/precision/f1: GNN retrieval metrics against target nodes
        - Configurable QA metrics (em, fm, f1, etc.)
        """
        if self.eval_graph_dataset_loader is None:
            return {}

        self.model.eval()

        # # Initialize metric collectors
        # all_metrics: dict[str, list[float]] = {
        #     "avg_selected_nodes": [],
        #     "gnn_recall": [],
        #     "gnn_precision": [],
        #     "gnn_f1": [],
        # }
        # # Add configured QA metrics
        # for metric_name in self.qa_metrics:
        #     all_metrics[metric_name] = []

        final_eval_metrics: dict[str, float] = {}

        for dataset in self.eval_graph_dataset_loader:
            task_dataset = self._create_task_dataset(dataset, is_train=False)
            dataset_name = task_dataset.name

            dataset_metrics: dict[str, list[float]] = {
                "avg_selected_nodes": []
            }

            predictions: list[dict[str, Any]] = []

            for metric_name in self.re_metrics:
                dataset_metrics[metric_name] = []

            for metric_name in self.qa_metrics:
                dataset_metrics[metric_name] = []

            graph = task_dataset.graph
            data_loader = task_dataset.data_loader
            node_texts = task_dataset.node_texts

            # Set epoch for sampler
            if hasattr(data_loader.sampler, "set_epoch"):
                data_loader.sampler.set_epoch(0)

            # Limit test samples if specified
            if self.args.max_eval_steps:
                data_iterator = islice(data_loader, self.args.max_eval_steps)
                total_steps = self.args.max_eval_steps
            else:
                data_iterator = data_loader
                total_steps = len(data_loader)

            # Get the model for generation (use base model, not parallel)
            model_for_gen = (
                self.model.module if hasattr(self.model, "module") else self.model
            )

            for batch in tqdm(
                data_iterator,
                desc=f"Evaluating {task_dataset.name}",
                total=total_steps,
                disable=utils.get_rank() != 0,
            ):
                batch = query_utils.cuda(batch, device=self.device)
                questions, answers = self._get_questions_and_answers(
                    batch, task_dataset.raw_data
                )
                sample_ids = batch["id"]

                # Generate answers and get retrieved nodes for metrics
                with torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=self.dtype,
                    enabled=self.use_amp,
                ):
                    generated_ids, retrieved_nodes = model_for_gen.generate(
                        graph=graph,
                        batch=batch,
                        questions=questions,
                        system_prompt=self.system_prompt,
                        node_texts=node_texts,
                        max_new_tokens=self.args.max_new_tokens,
                        threshold=self.args.threshold,
                        top_k=self.args.top_k,
                        max_graph_tokens=self.args.max_graph_tokens, #
                        return_retrieved_nodes=True,
                        do_sample=False,
                    )

                # Decode generated answers
                generated_texts = model_for_gen.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )

                # Build prediction records
                for idx, (sid, question, gold_answer, pred_text, retrieved) in enumerate(zip(
                    sample_ids, questions, answers, generated_texts, retrieved_nodes
                )):
                    # Compute QA metrics for this sample
                    sample_metrics = {}
                    for metric_name, metric_fn in self.qa_metrics.items():
                        sample_metrics[metric_name] = metric_fn(pred_text, gold_answer)
                    
                    # Compute GNN retrieval recall and precision using retrieved_nodes
                    target_doc_ids = graph.nodes_by_type['document']
                    target_mask = batch.get("target_nodes_mask", None)
                    target_mask = target_mask[idx, target_doc_ids]
                    if target_mask is not None:
                        num_nodes = target_mask.size(0)
                        target_mask_float = target_mask.float()
                        num_target = target_mask_float.sum(dim=-1)

                        for k in [2, 5, 10]:
                            pred_mask_k = torch.zeros(num_nodes, device=self.device)
                            for i, nodes in enumerate(retrieved_nodes):
                                top_k_nodes = nodes[:k]
                                if len(top_k_nodes) > 0:
                                    node_indices = torch.tensor(
                                        top_k_nodes, device=self.device, dtype=torch.long
                                    )
                                    valid_indices = node_indices[node_indices < num_nodes]
                                    if len(valid_indices) > 0:
                                        pred_mask_k[valid_indices] = 1.0
                            
                            tp_k = (pred_mask_k * target_mask_float).sum(dim=-1)
                            
                            recall_k = torch.where(
                                num_target > 0,
                                tp_k / num_target,
                                torch.zeros_like(tp_k),
                            )
                            sample_metrics[f"recall@{k}"] = recall_k.mean().item()
                    
                    retrieved_node_texts = (
                        [node_texts[idx] for idx in retrieved]
                        if retrieved is not None
                        else []
                    )
                    predictions.append(
                        {
                            "id": str(sid) if not isinstance(sid, str) else sid,
                            "question": question,
                            "answer": gold_answer,
                            "prediction": pred_text,
                            "retrieved_nodes": retrieved_node_texts,
                            "metrics": sample_metrics,
                        }
                    )
            
            # Gather predictions from all ranks if distributed
            if self.world_size > 1:
                import torch.distributed as dist

                # Gather all predictions to rank 0
                gathered_predictions = [None] * self.world_size
                dist.all_gather_object(gathered_predictions, predictions)

                if utils.get_rank() == 0:
                    # Flatten and deduplicate by ID
                    all_preds_flat = []
                    seen_ids = set()
                    for rank_preds in gathered_predictions:
                        for pred in rank_preds:
                            if pred["id"] not in seen_ids:
                                all_preds_flat.append(pred)
                                seen_ids.add(pred["id"])
                    predictions = all_preds_flat    
            
            agg_metrics = {}
            # calculate metrics
            if utils.get_rank() == 0 and len(predictions) > 0:
                # Compute aggregate metrics
                for metric_name in self.qa_metrics:
                    scores = [
                        p["metrics"].get(metric_name, 0.0) for p in predictions
                    ]
                    agg_metrics[f"{dataset_name}/{metric_name}"] = (
                        sum(scores) / len(scores) if scores else 0.0
                    )

                metric_strs = [
                    f"{name}: {score:.4f}"
                    for name, score in agg_metrics.items()
                ]
                logger.info(f"[{dataset_name}] QA Metrics: {', '.join(metric_strs)}")

                for metric_name in self.re_metrics:
                    scores = [
                        p["metrics"].get(metric_name, 0.0) for p in predictions
                    ]
                    agg_metrics[f"{dataset_name}/{metric_name}"] = (
                        sum(scores) / len(scores) if scores else 0.0
                    )

                metric_strs = [
                    f"{name}: {score:.4f}"
                    for name, score in agg_metrics.items() if name.split('/')[-1] in self.re_metrics
                ]
                logger.info(f"[{dataset_name}] Retrieval Metrics: {', '.join(metric_strs)}")

            final_eval_metrics.update(agg_metrics)
        
        return final_eval_metrics

    def _train_epoch(self) -> None:
        """Train for one epoch with temperature scheduling."""
        self.model.train()

        if self.train_graph_dataset_loader is None:
            return

        epoch_metrics: dict[str, list[float]] = {}

        for dataset in self.train_graph_dataset_loader:
            task_dataset = self._create_task_dataset(dataset, is_train=True)

            data_loader = task_dataset.data_loader

            # Set epoch for sampler
            if hasattr(data_loader.sampler, "set_epoch"):
                data_loader.sampler.set_epoch(self.state["epoch"])

            # Limit steps per epoch if specified
            if self.args.max_steps_per_epoch:
                data_iterator = islice(data_loader, self.args.max_steps_per_epoch)
                total_steps = self.args.max_steps_per_epoch
            else:
                data_iterator = data_loader
                total_steps = len(data_loader)

            pbar = tqdm(
                data_iterator,
                desc=f"Training {task_dataset.name}",
                total=total_steps,
                disable=not utils.is_main_process(),
            )
            for batch in pbar:
                self.state["global_step"] += 1

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward and backward with AMP
                with torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=self.dtype,
                    enabled=self.use_amp,
                ):
                    step_metrics = self.train_step(batch, task_dataset)

                loss = step_metrics["loss"]

                if self.state["global_step"] % 500 == 0 and utils.get_rank() == 0:
                    # 1. 针对你的 GFMLLMModel 精确提取核心权重参数
                    monitor_params = {
                        "GNN_Conv_0": self.model.graph_reasoner.entity_model.layers[0].linear.weight, 
                        "GNN_Conv_5": self.model.graph_reasoner.entity_model.layers[5].linear.weight, 
                        
                        # --- 图网络中层 (融合与非线性) ---
                        "GNN_Early_Fuse": self.model.graph_reasoner.early_fuse_mlp[0].weight,
                        
                        # --- 图网络顶层 (输出) ---
                        "GNN_Predict": self.model.graph_reasoner.predict_mlp[0].weight,
                        
                        # --- 跨模态桥梁 ---
                        "Proj_Layer": self.model.graph_projection[0].weight,
                        
                        # --- 大模型部分 (保留几个关键的作为对照) ---
                        "LLM_Embed": self.model.llm.model.embed_tokens.weight,
                    }

                    current_scale = self.scaler.get_scale()

                    # --- 测 LLM Loss (生成任务) 的梯度 ---
                    self.optimizer.zero_grad()
                    self.scaler.scale(step_metrics["llm_loss"]).backward(retain_graph=True)
                    for name, param in monitor_params.items():
                        if param.grad is not None:
                            real_grad = param.grad.norm().item() / current_scale
                            logger.info(f"[LLM Loss 产生] -> {name:12s}: {real_grad:.6e}")
                        else:
                            logger.info(f"[LLM Loss 产生] -> {name:12s}: None (无梯度流过)")

                    logger.info("-" * 45)

                    # --- 测 GNN Loss (检索/对齐任务) 的梯度 ---
                    self.optimizer.zero_grad()
                    # 注意这里带上了你联合训练的权重 alpha
                    self.scaler.scale(step_metrics["gnnn_loss"]).backward(retain_graph=True)
                    for name, param in monitor_params.items():
                        if param.grad is not None:
                            real_grad = param.grad.norm().item() / current_scale
                            logger.info(f"[GNN Loss 产生] -> {name:12s}: {real_grad:.6e}")
                        else:
                            logger.info(f"[GNN Loss 产生] -> {name:12s}: None (无梯度流过)")

                    logger.info("==================================================\n")
                    
                    # --- 测完清空，把干净的场子留给正式的 total_loss.backward() ---
                    self.optimizer.zero_grad()
                    
                # Backward pass
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.args.max_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.get_trainable_parameters(), # self.model.parameters(), 
                        self.args.max_grad_norm
                    )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.lr_scheduler.step()

                # Track metrics
                for key, value in step_metrics.items():
                    if key != "loss":
                        if key not in epoch_metrics:
                            epoch_metrics[key] = []
                        if isinstance(value, torch.Tensor):
                            value = value.item()
                        epoch_metrics[key].append(value)
                
                current_lr = self.optimizer.param_groups[0]["lr"]
                # Update progress bar
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "lm_loss": f"{step_metrics['lm_loss']:.4f}",
                        "gnn_loss": f"{step_metrics['gnn_loss']:.4f}",
                        "lr": f"{current_lr:.2e}",
                        "temp": f"{step_metrics.get('temperature', 0):.3f}",
                    }
                )

                # Evaluate by steps
                if (
                    self.eval_strategy == "steps"
                    and self.state["global_step"] % self.eval_steps == 0
                    and self.eval_graph_dataset_loader is not None
                ):
                    eval_metrics = self.evaluate()
                    self._log_metrics(eval_metrics, prefix="eval")
                    self._maybe_save_best_model(eval_metrics)
                    self.model.train()

        # Log epoch metrics
        if epoch_metrics:
            avg_metrics = {
                f"train_{key}": sum(values) / len(values)
                for key, values in epoch_metrics.items()
                if len(values) > 0
            }
            self._log_metrics(avg_metrics, prefix="train")

    def _maybe_save_best_model(self, eval_metrics: dict[str, float]) -> None:
        """Save model if it's the best so far."""
        metric_name = self.args.metric_for_best_model
        # if metric_name not in eval_metrics:
        #     return
        if any(metric not in eval_metrics for metric in metric_name):
            return

        # current_metric = eval_metrics[metric_name]
        current_metric = sum(eval_metrics[metric] for metric in metric_name) / len(metric_name)
        is_better = (
            current_metric > self.state["best_metric"]
            if self.args.greater_is_better
            else current_metric < self.state["best_metric"]
        )
        
        if utils.get_rank() == 0:
            logger.info(
                    f"Current Epoch {self.state['epoch']} average EM: {current_metric:.4f}"
                )

        if is_better:
            self.state["best_metric"] = current_metric
            self.state["best_epoch"] = self.state["epoch"]

            if utils.get_rank() == 0:
                self._save_checkpoint(self.output_dir, is_best=True)
                logger.info(
                    f"New best model! {metric_name}: {current_metric:.4f} at epoch {self.state['epoch']}"
                )

    def _get_trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Get state dict containing only trainable parameters.

        This saves space by not storing frozen pre-trained weights (GNN/LLM).

        Returns:
            Dict mapping parameter names to their values for trainable params only.
        """
        trainable_state_dict = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_state_dict[name] = param.data.clone()
        return trainable_state_dict

    def _save_checkpoint(self, output_dir: str, is_best: bool = False) -> None:
        """Save a checkpoint with only trainable parameters.

        Only saves parameters that have requires_grad=True, excluding frozen
        pre-trained weights from GNN and LLM to save disk space.

        Args:
            output_dir: Directory to save the checkpoint
            is_best: Whether this is the best model so far
        """
        if utils.get_rank() == 0:
            # Get only trainable parameters
            trainable_state_dict = self._get_trainable_state_dict()

            # Log which parameters are being saved
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(v.numel() for v in trainable_state_dict.values())
            logger.info(
                f"Saving {len(trainable_state_dict)} trainable parameter groups "
                f"({trainable_params:,} params out of {total_params:,} total, "
                f"{100 * trainable_params / total_params:.2f}%)"
            )

            state = {
                "trainable_model": trainable_state_dict,
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "epoch": self.state["epoch"],
                "global_step": self.state["global_step"],
                "best_metric": self.state["best_metric"],
                "best_epoch": self.state["best_epoch"],
                # Store list of trainable parameter names for verification during loading
                "trainable_param_names": list(trainable_state_dict.keys()),
            }

            # Save best checkpoint
            if is_best:
                best_path = os.path.join(output_dir, "model_best.pth")
                torch.save(state, best_path)
                logger.info(f"Saved best model to {best_path}")

                # Log model checkpoint to wandb
                log_model_checkpoint(
                    best_path,
                    f"best-epoch-{self.state['epoch']}-step-{self.state['global_step']}",
                    metadata={
                        "epoch": self.state["epoch"],
                        "best_metric": self.state["best_metric"],
                        "best": True,
                        "trainable_params": trainable_params,
                    },
                )
            # Save regular checkpoint
            elif not self.args.save_best_only:
                checkpoint_path = os.path.join(
                    output_dir,
                    f"checkpoint-epoch-{self.state['epoch']}-step-{self.state['global_step']}.pth",
                )
                torch.save(state, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
                log_model_checkpoint(
                    checkpoint_path,
                    f"checkpoint-epoch-{self.state['epoch']}-step-{self.state['global_step']}",
                    metadata={
                        "epoch": self.state["epoch"],
                        "global_step": self.state["global_step"],
                        "trainable_params": trainable_params,
                    },
                )

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a checkpoint with only trainable parameters.

        Loads the trainable parameters back into the model. The frozen pre-trained
        weights should already be loaded when the model is initialized.

        Args:
            checkpoint_path: Path to the checkpoint file
        """
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            state = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )

            # Check if this is a new-style checkpoint (trainable only) or old-style (full model)
            if "trainable_model" in state:
                # New-style checkpoint with only trainable parameters
                trainable_state_dict = state["trainable_model"]

                # Verify parameter names match current trainable parameters
                current_trainable = {
                    name
                    for name, param in self.model.named_parameters()
                    if param.requires_grad
                }
                saved_trainable = set(trainable_state_dict.keys())

                missing = current_trainable - saved_trainable
                unexpected = saved_trainable - current_trainable

                if missing:
                    logger.warning(
                        f"Missing trainable parameters in checkpoint: {missing}"
                    )
                if unexpected:
                    logger.warning(
                        f"Unexpected parameters in checkpoint (will be ignored): {unexpected}"
                    )

                # Load trainable parameters
                model_state_dict = self.model.state_dict()
                for name, param in trainable_state_dict.items():
                    if name in model_state_dict:
                        model_state_dict[name] = param
                    else:
                        logger.warning(f"Parameter {name} not found in model, skipping")

                self.model.load_state_dict(model_state_dict, strict=False)
                logger.info(
                    f"Loaded {len(trainable_state_dict)} trainable parameter groups"
                )

            elif "model" in state:
                # Old-style checkpoint with full model state
                logger.info("Loading old-style checkpoint with full model state")
                self.model.load_state_dict(state["model"], strict=False)

            # Load optimizer state
            if "optimizer" in state and hasattr(self, "optimizer"):
                try:
                    self.optimizer.load_state_dict(state["optimizer"])
                    logger.info("Loaded optimizer state from checkpoint")
                except Exception as e:
                    logger.warning(f"Could not load optimizer state: {e}")

            # Load training state
            if "epoch" in state:
                self.state["epoch"] = state["epoch"]
            if "global_step" in state:
                self.state["global_step"] = state["global_step"]
            if "best_metric" in state:
                self.state["best_metric"] = state["best_metric"]
            if "best_epoch" in state:
                self.state["best_epoch"] = state["best_epoch"]
            if "scaler" in state:
                self.scaler.load_state_dict(state["scaler"])

            logger.info(
                f"Resumed from epoch {self.state['epoch']}, global step {self.state['global_step']}"
            )
        else:
            logger.warning(f"Checkpoint {checkpoint_path} does not exist")

    @torch.no_grad()
    def predict(self) -> dict[str, list[dict[str, Any]]]:
        """Generate predictions on evaluation datasets.

        Generates answers for all samples in the evaluation datasets and returns
        structured prediction results including questions, generated answers,
        ground truth answers, and QA metrics.

        Returns:
            Dictionary mapping dataset names to lists of prediction dicts.
            Each prediction dict contains:
            - id: Sample ID
            - question: Original question
            - answer: Ground truth answer
            - prediction: Generated answer (full response)
            - extracted_prediction: Extracted answer from response
            - metrics: Dict of QA metric scores for this sample
        """
        if self.eval_graph_dataset_loader is None:
            logger.warning("No evaluation dataset loader provided for prediction")
            return {}

        self.model.eval()
        all_predictions: dict[str, list[dict[str, Any]]] = {}

        for dataset in self.eval_graph_dataset_loader:
            task_dataset = self._create_task_dataset(dataset, is_train=False)
            dataset_name = task_dataset.name

            # graph-bench prompt-bulider
            if 'graphrag' in dataset_name:
                from gfmrag.utils.gbench_prompt import load_prompt_builders
                prompt_builders = load_prompt_builders(self.prompt_map)

            graph = task_dataset.graph
            data_loader = task_dataset.data_loader
            node_texts = task_dataset.node_texts

            # Set epoch for sampler
            if hasattr(data_loader.sampler, "set_epoch"):
                data_loader.sampler.set_epoch(0)

            predictions: list[dict[str, Any]] = []

            # Get the model for generation
            model_for_gen = (
                self.model.module if hasattr(self.model, "module") else self.model
            )

            # Limit test samples if specified
            if self.args.max_eval_steps:
                data_iterator = islice(data_loader, self.args.max_eval_steps)
                total_steps = self.args.max_eval_steps
            else:
                data_iterator = data_loader
                total_steps = len(data_loader)

            for batch in tqdm(
                data_iterator,
                desc=f"Predicting {dataset_name}",
                total=total_steps,
                disable=utils.get_rank() != 0,
            ):
                batch = query_utils.cuda(batch, device=self.device)

                questions, answers = self._get_questions_and_answers(
                    batch, task_dataset.raw_data
                )
                sample_ids = batch["id"]
                
                if dataset_name == 'graphrag_bench_cs':
                    task_types = [id.split('-')[0] for id in sample_ids]
                    self.system_prompt = prompt_builders[task_types[0]].system_prompt
                elif dataset_name == 'graphrag_benchmark_novel' or dataset_name == 'graphrag_benchmark_medical':
                    task_types = batch['question_type']
                    self.system_prompt = prompt_builders[task_types[0]].system_prompt
                
                # Generate answers
                with torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=self.dtype,
                    enabled=self.use_amp,
                ):
                    generated_ids, retrieved_nodes = model_for_gen.generate(
                        graph=graph,
                        batch=batch,
                        questions=questions,
                        system_prompt=self.system_prompt,
                        node_texts=node_texts,
                        max_new_tokens=self.args.max_new_tokens,
                        threshold=self.args.threshold,
                        top_k=self.args.top_k,
                        max_graph_tokens=self.args.max_graph_tokens,
                        return_retrieved_nodes=True,
                        do_sample=False,
                    )

                # Decode generated answers
                generated_texts = model_for_gen.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )

                # Build prediction records
                for sid, question, gold_answer, pred_text, retrieved in zip(
                    sample_ids, questions, answers, generated_texts, retrieved_nodes
                ):
                    # Compute QA metrics for this sample
                    sample_metrics = {}
                    for metric_name, metric_fn in self.qa_metrics.items():
                        sample_metrics[metric_name] = metric_fn(pred_text, gold_answer)

                    retrieved_node_texts = (
                        [node_texts[idx] for idx in retrieved]
                        if retrieved is not None
                        else []
                    )
                    predictions.append(
                        {
                            "id": str(sid) if not isinstance(sid, str) else sid,
                            "question": question,
                            "answer": gold_answer,
                            "prediction": pred_text,
                            "retrieved_nodes": retrieved_node_texts,
                            "metrics": sample_metrics,
                        }
                    )

            # Gather predictions from all ranks if distributed
            if self.world_size > 1:
                import torch.distributed as dist

                # Gather all predictions to rank 0
                gathered_predictions = [None] * self.world_size
                dist.all_gather_object(gathered_predictions, predictions)

                if utils.get_rank() == 0:
                    # Flatten and deduplicate by ID
                    all_preds_flat = []
                    seen_ids = set()
                    for rank_preds in gathered_predictions:
                        for pred in rank_preds:
                            if pred["id"] not in seen_ids:
                                all_preds_flat.append(pred)
                                seen_ids.add(pred["id"])
                    predictions = all_preds_flat

            all_predictions[dataset_name] = predictions

            # Log summary metrics for this dataset
            if utils.get_rank() == 0 and len(predictions) > 0:
                logger.info(
                    f"Generated {len(predictions)} predictions for {dataset_name}"
                )

                # Compute aggregate metrics
                if self.qa_metrics:
                    agg_metrics = {}
                    for metric_name in self.qa_metrics:
                        scores = [
                            p["metrics"].get(metric_name, 0.0) for p in predictions
                        ]
                        agg_metrics[metric_name] = (
                            sum(scores) / len(scores) if scores else 0.0
                        )

                    metric_strs = [
                        f"{name.upper()}: {score:.4f}"
                        for name, score in agg_metrics.items()
                    ]
                    logger.info(f"  Metrics: {', '.join(metric_strs)}")

        return all_predictions

'''
@torch.no_grad()
    def predict(self) -> dict[str, list[dict[str, Any]]]:
        """Generate predictions on evaluation datasets.

        Generates answers for all samples in the evaluation datasets and returns
        structured prediction results including questions, generated answers,
        ground truth answers, and QA metrics.

        Returns:
            Dictionary mapping dataset names to lists of prediction dicts.
            Each prediction dict contains:
            - id: Sample ID
            - question: Original question
            - answer: Ground truth answer
            - prediction: Generated answer (full response)
            - extracted_prediction: Extracted answer from response
            - metrics: Dict of QA metric scores for this sample
        """
        if self.eval_graph_dataset_loader is None:
            logger.warning("No evaluation dataset loader provided for prediction")
            return {}

        self.model.eval()
        all_predictions: dict[str, list[dict[str, Any]]] = {}

        for dataset in self.eval_graph_dataset_loader:
            task_dataset = self._create_task_dataset(dataset, is_train=False)
            dataset_name = task_dataset.name

            graph = task_dataset.graph
            data_loader = task_dataset.data_loader
            node_texts = task_dataset.node_texts

            # Set epoch for sampler
            if hasattr(data_loader.sampler, "set_epoch"):
                data_loader.sampler.set_epoch(0)

            predictions: list[dict[str, Any]] = []

            # Get the model for generation
            model_for_gen = (
                self.model.module if hasattr(self.model, "module") else self.model
            )

            # Limit test samples if specified
            if self.args.max_eval_steps:
                data_iterator = islice(data_loader, self.args.max_eval_steps)
                total_steps = self.args.max_eval_steps
            else:
                data_iterator = data_loader
                total_steps = len(data_loader)

            for batch in tqdm(
                data_iterator,
                desc=f"Predicting {dataset_name}",
                total=total_steps,
                disable=utils.get_rank() != 0,
            ):
                batch = query_utils.cuda(batch, device=self.device)

                questions, answers = self._get_questions_and_answers(
                    batch, task_dataset.raw_data
                )
                sample_ids = batch["id"]

                # Generate answers
                with torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=self.dtype,
                    enabled=self.use_amp,
                ):
                    generated_ids, retrieved_nodes = model_for_gen.generate(
                        graph=graph,
                        batch=batch,
                        questions=questions,
                        system_prompt=self.system_prompt,
                        node_texts=node_texts,
                        max_new_tokens=self.args.max_new_tokens,
                        threshold=self.args.threshold,
                        top_k=self.args.top_k,
                        max_graph_tokens=self.args.max_graph_tokens,
                        return_retrieved_nodes=True,
                        do_sample=False,
                    )

                # Decode generated answers
                generated_texts = model_for_gen.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )

                # Build prediction records
                for sid, question, gold_answer, pred_text, retrieved in zip(
                    sample_ids, questions, answers, generated_texts, retrieved_nodes
                ):
                    # Compute QA metrics for this sample
                    sample_metrics = {}
                    for metric_name, metric_fn in self.qa_metrics.items():
                        sample_metrics[metric_name] = metric_fn(pred_text, gold_answer)

                    retrieved_node_texts = (
                        [node_texts[idx] for idx in retrieved]
                        if retrieved is not None
                        else []
                    )
                    predictions.append(
                        {
                            "id": str(sid) if not isinstance(sid, str) else sid,
                            "question": question,
                            "answer": gold_answer,
                            "prediction": pred_text,
                            "retrieved_nodes": retrieved_node_texts,
                            "metrics": sample_metrics,
                        }
                    )

            # Gather predictions from all ranks if distributed
            if self.world_size > 1:
                import torch.distributed as dist

                # Gather all predictions to rank 0
                gathered_predictions = [None] * self.world_size
                dist.all_gather_object(gathered_predictions, predictions)

                if utils.get_rank() == 0:
                    # Flatten and deduplicate by ID
                    all_preds_flat = []
                    seen_ids = set()
                    for rank_preds in gathered_predictions:
                        for pred in rank_preds:
                            if pred["id"] not in seen_ids:
                                all_preds_flat.append(pred)
                                seen_ids.add(pred["id"])
                    predictions = all_preds_flat

            all_predictions[dataset_name] = predictions

            # Log summary metrics for this dataset
            if utils.get_rank() == 0 and len(predictions) > 0:
                logger.info(
                    f"Generated {len(predictions)} predictions for {dataset_name}"
                )

                # Compute aggregate metrics
                if self.qa_metrics:
                    agg_metrics = {}
                    for metric_name in self.qa_metrics:
                        scores = [
                            p["metrics"].get(metric_name, 0.0) for p in predictions
                        ]
                        agg_metrics[metric_name] = (
                            sum(scores) / len(scores) if scores else 0.0
                        )

                    metric_strs = [
                        f"{name.upper()}: {score:.4f}"
                        for name, score in agg_metrics.items()
                    ]
                    logger.info(f"  Metrics: {', '.join(metric_strs)}")

        return all_predictions

'''