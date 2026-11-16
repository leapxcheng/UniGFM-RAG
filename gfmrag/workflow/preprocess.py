import os
import logging
import torch
import torch.distributed as dist
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf

# 假设你的 utils 函数在 gfmrag 包中
from gfmrag import utils

logger = logging.getLogger(__name__)

@hydra.main(
    config_path="config/gfm_llm", 
    config_name="index_dataset", # 对应你的 YAML 文件名 (不含后缀)
    version_base=None
)
def main(cfg: DictConfig) -> None:
    current_rank = utils.get_rank()

    if current_rank == 0:
        output_dir = HydraConfig.get().runtime.output_dir
        logger.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Output directory: {output_dir}")
        output_dir_list = [output_dir]
    else:
        output_dir_list = [None]

    train_names = cfg.get("train_names", [])
    valid_names = cfg.get("valid_names", [])
    all_data_names = list(set(list(train_names) + list(valid_names)))
    print(all_data_names)

    for dataset_name in all_data_names:
        dataset = instantiate(
            {"_target_": cfg.datasets._target_},
            **cfg.datasets,
            data_name=dataset_name,
            _recursive_=False
        )

if __name__ == "__main__":
    main()