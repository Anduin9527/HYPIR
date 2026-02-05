# -*- coding: utf-8 -*-
"""
isBlur 分类器：用于识别模糊图像的轻量级二分类模型
"""
import torch
import torch.nn as nn
import torchvision.models as models


class BlurClassifier(nn.Module):
    """轻量级模糊检测分类器
    
    使用预训练的 CNN backbone（ResNet18 或 EfficientNet-B0）进行二分类
    输入：[-1, 1] 范围的图像（HYPIR 格式）
    输出：logits，用于 BCE loss
    """
    
    def __init__(self, backbone='resnet18', pretrained=True):
        """
        Args:
            backbone: 'resnet18' 或 'efficientnet_b0'
            pretrained: 是否使用 ImageNet 预训练权重
        """
        super().__init__()
        
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            # 修改最后一层为二分类（1个输出）
            self.backbone.fc = nn.Linear(512, 1)
            
        elif backbone == 'efficientnet_b0':
            from torchvision.models import efficientnet_b0
            self.backbone = efficientnet_b0(pretrained=pretrained)
            # EfficientNet 的分类层在 classifier[1]
            self.backbone.classifier[1] = nn.Linear(1280, 1)
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose 'resnet18' or 'efficientnet_b0'")
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W], range [-1, 1] (HYPIR format)
        
        Returns:
            logits: [B, 1], raw logits for BCE loss
        """
        # 转换到 [0, 1] range (ImageNet pretrained models expect this)
        x = (x + 1) / 2
        return self.backbone(x)


if __name__ == "__main__":
    # 简单测试
    model = BlurClassifier(backbone='resnet18', pretrained=False)
    x = torch.randn(4, 3, 224, 224) * 2 - 1  # [-1, 1] range
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Logits: {logits.squeeze().tolist()}")
    
    # 测试 sigmoid
    probs = torch.sigmoid(logits)
    print(f"Probabilities: {probs.squeeze().tolist()}")
