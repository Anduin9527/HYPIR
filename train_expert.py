# -*- coding: utf-8 -*-
"""
阶段 1a：预告训练 NAFNet Expert
用于训练不同退化的专家网络 (Rain, Blur, Lowlight, Snow, Haze)
"""

from argparse import ArgumentParser
from omegaconf import OmegaConf
from HYPIR.trainer.expert_trainer import ExpertTrainer


def main():
    parser = ArgumentParser(description="训练单个 NAFNet Expert 模块")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="配置文件路径 (通常指向 configs/nafnet_expert.yaml)",
    )
    parser.add_argument(
        "--expert_type",
        type=str,
        required=True,
        choices=["Rain", "Blur", "Lowlight", "Snow", "Haze"],
        help="当前训练的专家类型，用于过滤数据集",
    )
    args = parser.parse_args()

    # 加载配置
    config = OmegaConf.load(args.config)

    # 动态将命令行参数的 expert_type 合并到 config 中
    OmegaConf.set_struct(config, False)
    config.expert_type = args.expert_type

    # 创建训练器并训练
    trainer = ExpertTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
