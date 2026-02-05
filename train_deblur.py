# -*- coding: utf-8 -*-
"""
阶段 1b：微调 NAFNet 去模糊模块
冻结 isBlur，在 Blur 数据上微调 NAFNet
"""
from argparse import ArgumentParser
from omegaconf import OmegaConf
from HYPIR.trainer.deblur_trainer import DeblurTrainer


def main():
    parser = ArgumentParser(description="微调 NAFNet 去模糊模块")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    args = parser.parse_args()
    
    # 加载配置
    config = OmegaConf.load(args.config)
    
    # 创建训练器并训练
    trainer = DeblurTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
