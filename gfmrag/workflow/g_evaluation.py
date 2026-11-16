"""Workflow for GFM-LLM end-to-end training with QA evaluation."""

import json
import logging
import os

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import distributed as dist

from gfmrag import utils
from gfmrag.graph_index_datasets import GraphDatasetLoader
from gfmrag.utils.wandb_utils import (
    finish_wandb,
    init_wandb,
    watch_model,
)

# A logger for this file
logger = logging.getLogger(__name__)

separator = ">" * 30
line = "-" * 30


@hydra.main(
    config_path="config/gfm_llm", config_name="gfm_llm_training", version_base=None
)
def main(cfg: DictConfig) -> None:
    utils.init_distributed_mode(cfg.timeout)
    torch.manual_seed(cfg.seed + utils.get_rank())

    if utils.get_rank() == 0:
        output_dir = HydraConfig.get().runtime.output_dir
        logger.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Output directory: {output_dir}")
        output_dir_list = [output_dir]
    else:
        output_dir_list = [None]

    if utils.get_world_size() > 1:
        dist.broadcast_object_list(output_dir_list, src=0)
    output_dir = output_dir_list[0]

    # Initialize the datasets in each process
    if cfg.datasets.init_datasets:
        feat_dim_list = utils.init_multi_dataset(
            cfg, utils.get_world_size(), utils.get_rank()
        )
        feat_dim = set(feat_dim_list)
        assert len(feat_dim) == 1, (
            "All datasets should have the same feature embedding dimension"
        )
    else:
        assert cfg.datasets.feat_dim is not None, (
            "If datasets.init_datasets is False, cfg.datasets.feat_dim must be set"
        )
        feat_dim = {cfg.datasets.feat_dim}

    if utils.get_rank() == 0:
        logger.info(
            f"Datasets {cfg.datasets.train_names} and {cfg.datasets.valid_names} initialized"
        )

    # Load model from pre-trained format or instantiate from config
    if cfg.load_model_from_pretrained:
        from gfmrag.models.gfm_llm import GFMLLMModel

        model = GFMLLMModel.from_pretrained(cfg.load_model_from_pretrained)
        logger.info(
            f"Loaded pre-trained GFM-LLM model from {cfg.load_model_from_pretrained}"
        )
    else:
        model = instantiate(cfg.model, feat_dim=feat_dim.pop())

    if utils.get_rank() == 0:
        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(line)
        logger.info(f"Total parameters: {num_params:,}")
        logger.info(
            f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / num_params:.2f}%)"
        )

    # Create data loaders
    train_graph_dataset_loader = GraphDatasetLoader(
        cfg.datasets,
        cfg.datasets.train_names,
        max_datasets_in_memory=cfg.datasets.max_datasets_in_memory,
        data_loading_workers=cfg.datasets.data_loading_workers,
    )
    valid_graph_dataset_loader = GraphDatasetLoader(
        cfg.datasets,
        cfg.datasets.valid_names,
        shuffle=False,
        max_datasets_in_memory=cfg.datasets.max_datasets_in_memory,
        data_loading_workers=cfg.datasets.data_loading_workers,
    )

    # different module
    llm_params = [p for p in model.llm.parameters() if p.requires_grad]
    gnn_params = [p for p in model.graph_reasoner.parameters() if p.requires_grad]
    proj_params = [p for p in model.graph_projection.parameters() if p.requires_grad]
    gnn_lr = cfg.module_lrs.gnn
    proj_lr = cfg.module_lrs.proj

    optimizer_grouped_parameters = [
        {
            "params": llm_params,
            "name": "llm_params"
        },
        {
            "params": gnn_params,
            "lr": gnn_lr,
            "name": "gnn_params"
        },
        {
            "params": proj_params,
            "lr": proj_lr,
            "name": "proj_params"
        }
    ]
    optimizer = instantiate(
        cfg.optimizer, 
        optimizer_grouped_parameters
    )

    # Initialize learning rate scheduler if configured
    lr_scheduler = None
    if hasattr(cfg, "lr_scheduler") and cfg.lr_scheduler is not None:
        # Calculate total training steps for scheduler
        total_steps = cfg.trainer.args.num_epoch * (
            cfg.trainer.args.max_steps_per_epoch or 10000
        )
        lr_scheduler = instantiate(
            cfg.lr_scheduler,
            optimizer=optimizer,
            num_training_steps=total_steps,
        )

    # Initialize GNN loss function if configured
    gnn_loss_fn = None
    if hasattr(cfg, "gnn_loss") and cfg.gnn_loss is not None:
        gnn_loss_fn = instantiate(cfg.gnn_loss)

    # Get QA metrics from config (default: ["em", "fm", "f1"])
    qa_metrics = None
    if hasattr(cfg.trainer, "qa_metrics") and cfg.trainer.qa_metrics is not None:
        qa_metrics = list(cfg.trainer.qa_metrics)
    
    # Get retrieval metrics from config (default: ["Recall@2", "Recall@5", "Recall@10"])
    re_metrics = None
    if hasattr(cfg.trainer, "re_metrics") and cfg.trainer.re_metrics is not None:
        re_metrics = list(cfg.trainer.re_metrics)
    
    # cs
    prompt_map = None
    if 'graphrag_bench_cs' in cfg.datasets.valid_names:
        if hasattr(cfg.trainer, "cs_prompt_map") and cfg.trainer.cs_prompt_map is not None:
            prompt_map = cfg.trainer.cs_prompt_map
    else:
        # medical/novel
        if hasattr(cfg.trainer, "prompt_map") and cfg.trainer.prompt_map is not None:
            prompt_map = cfg.trainer.prompt_map
    
    # Create trainer
    trainer = instantiate(
        cfg.trainer,
        output_dir=output_dir,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        gnn_loss_fn=gnn_loss_fn,
        qa_metrics=qa_metrics,
        re_metrics=re_metrics,
        prompt_map=prompt_map,
        train_graph_dataset_loader=train_graph_dataset_loader,
        eval_graph_dataset_loader=valid_graph_dataset_loader,
    )

    # Train
    trainer.train()

    # Save the model into the format for inference
    if utils.is_main_process() and cfg.save_pretrained:
        pre_trained_dir = os.path.join(output_dir, "pretrained")
        model.save_pretrained(pre_trained_dir)
        logger.info(f"Model saved to {pre_trained_dir}")

    # Generate predictions if requested
    if trainer.args.do_predict:
        predictions = trainer.predict()
        if utils.is_main_process():
            for data_name, preds in predictions.items():
                pred_path = os.path.join(output_dir, f"predictions_{data_name}.json")
                with open(pred_path, "w") as f:
                    json.dump(preds, f, indent=4)
                logger.info(f"Predictions saved to {pred_path}")

    # Shutdown the dataset loaders
    train_graph_dataset_loader.shutdown()
    valid_graph_dataset_loader.shutdown()

    utils.synchronize()
    utils.cleanup()

    # Finish wandb logging
    if utils.get_rank() == 0:
        finish_wandb()


if __name__ == "__main__":
    main()
