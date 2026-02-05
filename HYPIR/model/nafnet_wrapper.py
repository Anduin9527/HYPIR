# -*- coding: utf-8 -*-
"""
NAFNet 包装器：处理 HYPIR 数据格式和归一化
使用本地的 NAFNet 实现，无外部依赖
"""
import torch
import torch.nn as nn
from pathlib import Path

# 从本地导入 NAFNet
from HYPIR.model.nafnet import NAFNet


class NAFNetWrapper(nn.Module):
    """NAFNet 包装器，处理 HYPIR 数据格式
    
    主要功能：
    1. 使用本地 NAFNet 实现（无外部依赖）
    2. 处理归一化：HYPIR 使用 [-1, 1]，NAFNet 使用 [0, 1]
    3. 加载预训练权重
    4. 支持冻结/解冻
    """
    
    def __init__(self, checkpoint_path=None, width=64, enc_blks=None, 
                 middle_blk_num=12, dec_blks=None, freeze=False):
        """
        Args:
            checkpoint_path: NAFNet 预训练权重路径
            width: NAFNet 宽度（通道数）
            enc_blks: Encoder blocks 配置
            middle_blk_num: Middle blocks 数量
            dec_blks: Decoder blocks 配置
            freeze: 是否冻结参数
        """
        super().__init__()
        
        # 默认配置（与 GoPro 预训练模型一致）
        if enc_blks is None:
            enc_blks = [2, 2, 4, 8]
        if dec_blks is None:
            dec_blks = [2, 2, 2, 2]
        
        # 创建 NAFNet 模型（使用本地实现）
        print(f"创建 NAFNet: width={width}, enc_blks={enc_blks}, middle={middle_blk_num}, dec_blks={dec_blks}")
        self.nafnet = NAFNet(
            img_channel=3,
            width=width,
            middle_blk_num=middle_blk_num,
            enc_blk_nums=enc_blks,
            dec_blk_nums=dec_blks
        )
        
        # 加载预训练权重
        if checkpoint_path and Path(checkpoint_path).exists():
            self._load_checkpoint(checkpoint_path)
        
        # 冻结参数（用于阶段 2）
        if freeze:
            self.nafnet.eval()
            for param in self.nafnet.parameters():
                param.requires_grad = False
    
    def _load_checkpoint(self, checkpoint_path):
        """加载预训练权重，处理多种 checkpoint 格式"""
        state = torch.load(checkpoint_path, map_location='cpu')
        
        # 处理不同的 checkpoint 格式
        if isinstance(state, dict):
            if 'params' in state:
                state_dict = state['params']
            elif 'state_dict' in state:
                state_dict = state['state_dict']
            elif 'model' in state:
                state_dict = state['model']
            else:
                state_dict = state
        else:
            state_dict = state
        
        # 加载权重
        missing, unexpected = self.nafnet.load_state_dict(state_dict, strict=False)
        
        if missing:
            print(f"⚠️  NAFNet 加载权重时缺少的键: {missing[:5]}...")  # 只显示前5个
        if unexpected:
            print(f"⚠️  NAFNet 加载权重时多余的键: {unexpected[:5]}...")
        
        print(f"✅ 从 {checkpoint_path} 加载 NAFNet 权重")
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W], range [-1, 1] (HYPIR format)
        
        Returns:
            out: [B, 3, H, W], range [-1, 1]
        """
        # NAFNet 期望 [0, 1] range
        x_01 = (x + 1) / 2
        
        # NAFNet 前向传播
        out_01 = self.nafnet(x_01)
        
        # 转换回 [-1, 1] range
        return out_01 * 2 - 1


if __name__ == "__main__":
    # 测试代码
    print("测试 NAFNetWrapper...")
    
    # 创建模型（不加载权重）
    model = NAFNetWrapper(
        checkpoint_path=None,
        width=64,
        freeze=False
    )
    
    # 测试前向传播
    x = torch.randn(2, 3, 256, 256) * 2 - 1  # [-1, 1] range
    print(f"输入形状: {x.shape}, 范围: [{x.min():.2f}, {x.max():.2f}]")
    
    with torch.no_grad():
        out = model(x)
    
    print(f"输出形状: {out.shape}, 范围: [{out.min():.2f}, {out.max():.2f}]")
    
    # 计算参数量
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"参数量: {params:.2f}M")
    
    # 测试冻结
    model_frozen = NAFNetWrapper(checkpoint_path=None, freeze=True)
    trainable = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)
    print(f"冻结后可训练参数: {trainable}")
