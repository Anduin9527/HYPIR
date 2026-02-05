from argparse import ArgumentParser
from omegaconf import OmegaConf
from pathlib import Path

from HYPIR.trainer.sd2 import SD2Trainer


parser = ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()
config = OmegaConf.load(args.config)

# 添加配置文件名到config中，用于SwanLab实验命名
config_filename = Path(args.config).stem  # 获取文件名（不含扩展名）
config.config_name = config_filename

if config.base_model_type == "sd2":
    trainer = SD2Trainer(config)
    trainer.run()
elif config.base_model_type == "cascade_sd2":
    from HYPIR.trainer.cascade_sd2 import CascadeSD2Trainer
    trainer = CascadeSD2Trainer(config)
    trainer.run()
else:
    raise ValueError(f"Unsupported model type: {config.base_model_type}")
