# -*- coding: utf-8 -*-
"""
阶段：微调 NAFNet+MOE 一体式图像修复模块
冻结 5 个预训练的 NAFNet Experts，仅仅训练 Router。
"""

from argparse import ArgumentParser
from omegaconf import OmegaConf
from HYPIR.trainer.moe_trainer import MoeTrainer


def main():
    parser = ArgumentParser(description="微调 NAFNet MOE 路由模块")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="配置文件路径 (通常指向 configs/nafnet_moe.yaml)",
    )
    args = parser.parse_args()

    # 加载配置
    config = OmegaConf.load(args.config)

    # 创建训练器并训练
    trainer = MoeTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
