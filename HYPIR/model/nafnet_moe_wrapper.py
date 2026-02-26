# -*- coding: utf-8 -*-
"""
NAFNet MOE 包装器
处理 HYPIR 数据格式 ([-1, 1]) 与 NAFNet 数据格式 ([0, 1]) 之间的转换
"""

import torch
import torch.nn as nn
from pathlib import Path

from HYPIR.model.nafnet_moe import NAFNetMOE


class NAFNetMOEWrapper(nn.Module):
    """
    包装 NAFNetMOE，处理输入输出的数据范围规范
    HYPIR 默认使用 [-1, 1]
    NAFNet 默认使用 [0, 1]
    """

    def __init__(
        self,
        checkpoint_path=None,
        num_experts=5,
        router_channels=32,
        width=64,
        enc_blks=[2, 2, 4, 8],
        middle_blk_num=12,
        dec_blks=[2, 2, 2, 2],
        freeze=True,
    ):
        super().__init__()

        self.moe = NAFNetMOE(
            num_experts=num_experts,
            router_base_channels=router_channels,
            img_channel=3,
            width=width,
            middle_blk_num=middle_blk_num,
            enc_blk_nums=enc_blks,
            dec_blk_nums=dec_blks,
        )

        if checkpoint_path and Path(checkpoint_path).exists():
            self._load_checkpoint(checkpoint_path)

        if freeze:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False

    def _load_checkpoint(self, checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(state, dict):
            if "moe_net" in state:
                state_dict = state["moe_net"]
            elif "state_dict" in state:
                state_dict = state["state_dict"]
            else:
                state_dict = state
        else:
            state_dict = state

        missing, unexpected = self.moe.load_state_dict(state_dict, strict=False)
        print(f"✅ 从 {checkpoint_path} 加载 NAFNet+MOE 权重")

    def forward(self, x, top_k=None, return_routing=False):
        """
        x: [B, 3, H, W] in [-1, 1]
        """
        # 转换到 [0, 1]
        x_01 = (x + 1) / 2

        # 前向传播
        if return_routing:
            out_01, bal_loss, weights = self.moe(x_01, top_k=top_k, return_routing=True)
            out_neg1_1 = out_01 * 2 - 1
            return out_neg1_1, bal_loss, weights
        else:
            out_01, _ = self.moe(x_01, top_k=top_k, return_routing=False)
            out_neg1_1 = out_01 * 2 - 1
            return out_neg1_1
