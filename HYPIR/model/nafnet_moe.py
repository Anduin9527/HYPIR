# -*- coding: utf-8 -*-
"""
NAFNet + MOE 一体式底层视觉预处理模块
结合 5 个 NAFNet Experts 与 1 个轻量级 Image-level Router
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 从当前目录相对导入 NAFNet
from .nafnet import NAFNet


class LightRouter(nn.Module):
    """
    轻量级的 Image-level Router
    输入: [B, C, H, W] 图像
    输出: [B, num_experts] 的软路由权重 (Softmax后)
    """

    def __init__(self, in_channels=3, num_experts=5, base_channels=32):
        super().__init__()

        # 简单高效的卷积特征提取
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels, base_channels, kernel_size=3, stride=2, padding=1
            ),  # H/2
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1
            ),  # H/4
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1
            ),  # H/8
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                base_channels * 4, base_channels * 4, kernel_size=3, stride=2, padding=1
            ),  # H/16
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 全局池化降至 1x1
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 降维映射到专家数量
        self.fc = nn.Sequential(
            nn.Linear(base_channels * 4, base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            # 也可以在这里加上 Dropout 防止过拟合
            nn.Linear(base_channels, num_experts),
        )

    def forward(self, x):
        """
        返回未经 Softmax 的 logits 和 Softmax 后的 routing_weights
        """
        feat = self.features(x)  # [B, C*4, H/16, W/16]
        feat = self.avg_pool(feat)  # [B, C*4, 1, 1]
        feat = feat.view(feat.size(0), -1)  # [B, C*4]

        logits = self.fc(feat)  # [B, num_experts]
        routing_weights = F.softmax(logits, dim=1)  # [B, num_experts]

        return logits, routing_weights


class NAFNetMOE(nn.Module):
    """
    Network-Level MOE 架构，集成 Router 和多个 NAFNet 专家
    """

    def __init__(
        self,
        num_experts=5,
        router_base_channels=32,
        # 下面是 NAFNet 共享的架构参数
        img_channel=3,
        width=32,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1, 28],
        dec_blk_nums=[1, 1, 1, 1],
    ):
        super().__init__()
        self.num_experts = num_experts

        # 初始化 Router
        self.router = LightRouter(
            in_channels=img_channel,
            num_experts=num_experts,
            base_channels=router_base_channels,
        )

        # 初始化 5 个 NAFNet Experts
        # 使用 ModuleList 包裹所有的 Expert
        self.experts = nn.ModuleList(
            [
                NAFNet(
                    img_channel=img_channel,
                    width=width,
                    middle_blk_num=middle_blk_num,
                    enc_blk_nums=enc_blk_nums,
                    dec_blk_nums=dec_blk_nums,
                )
                for _ in range(num_experts)
            ]
        )

    def freeze_experts(self):
        """冻结所有的 NAFNet 参数，仅供 Router 训练"""
        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = False

    def unfreeze_experts(self):
        """解冻 NAFNet 参数，允许联合微调"""
        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = True

    def _calc_load_balance_loss(self, routing_weights):
        """
        计算负载均衡损失 (CV : Coefficient of Variation_squared)
        用来惩罚 Router 总是只选同一个专家 (专家崩塌)
        routing_weights: [B, num_experts]
        """
        # 计算当前 batch 内，每个专家的平均分配比例 -> f_i
        # f_i 的期望理想值应该是 1/num_experts
        mean_routing_weights = routing_weights.mean(dim=0)  # [num_experts]

        # 计算平衡损失：均方误差的变体，f_i * 专家数量。
        # 当完全均匀分布时 (每个都是 1/num_experts)，结果为 1。偏离越远越大。
        # 最常见的 MOE Load Balance 计算公式：
        num_experts = self.num_experts
        load_balance_loss = num_experts * torch.sum(mean_routing_weights**2)

        # 这里返回的多余部分（减掉理想值 1），确保极小值为 0
        return load_balance_loss - 1.0

    def forward(self, x, top_k=None, return_routing=False):
        """
        提供 top_k (int) 将在推理时开启硬路由（只传给概率最大的前 k 个专家），以节省时间显存。
        由于目前需要并行计算梯度，通常 training 阶段保持 top_k=None 以全局加权融合。
        """
        B, C, H, W = x.shape

        # 1. 路由分配
        logits, routing_weights = self.router(x)  # [B, num_experts]

        # 计算负载均衡损失 (记录仅用于训练期监督)
        balance_loss = self._calc_load_balance_loss(routing_weights)

        # 2. 如果开启 Top-K 硬路由过滤
        if top_k is not None and top_k < self.num_experts:
            # 取最大前 K 个的权重和索引
            top_weights, top_indices = torch.topk(routing_weights, top_k, dim=1)

            # 由于裁剪了部分内容，将前 K 个权重重新 softmax 归一化 (或直接除以它们的和)
            top_weights = top_weights / (top_weights.sum(dim=1, keepdim=True) + 1e-8)

            # 初始化输出
            out = torch.zeros_like(x)

            # 对 Batch 内每一个样本，循环查找命中的 expert 并累加
            for batch_idx in range(B):
                for k in range(top_k):
                    exp_idx = top_indices[batch_idx, k].item()
                    w = top_weights[batch_idx, k]

                    # 取出当前 sample 并送入 expert，升档回 [1,C,H,W]
                    sample_x = x[batch_idx].unsqueeze(0)
                    expert_y = self.experts[exp_idx](sample_x)

                    # 加上该专家的贡献
                    out[batch_idx] += expert_y.squeeze(0) * w

        else:
            # 走常规的完全加权融合 (全通 Soft Routing)
            # 因为只有 5 个 NAFNet，这里用显存换连续反向传播空间
            expert_outputs = []
            for i in range(self.num_experts):
                # 过该专家前向
                out_i = self.experts[i](x)  # [B, C, H, W]
                # 加权：w_i is [B], 扩展成 [B, 1, 1, 1] 方便广播相乘
                w_i = routing_weights[:, i].view(B, 1, 1, 1)
                expert_outputs.append(out_i * w_i)

            # 直接累加
            out = sum(expert_outputs)

        if return_routing:
            return out, balance_loss, routing_weights

        return out, balance_loss


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("Testing NAFNetMOE...")

    # 构建测试用的迷尔模型以防显存爆炸
    net = NAFNetMOE(
        num_experts=5,
        width=16,  # 测试用的小 NAFNet
        middle_blk_num=1,
        enc_blk_nums=[1, 1],
        dec_blk_nums=[1, 1],
    )

    # 模拟输入
    x = torch.randn(2, 3, 256, 256)
    print("Input shape: {}".format(x.shape))

    # 测试全路由前向
    y, bal_loss, weights = net(x, return_routing=True)
    print("--- Full Routing ---")
    print("Output shape: {}".format(y.shape))
    print("Balance loss: {:.4f}".format(bal_loss.item()))
    print("Weights shape: {}".format(weights.shape))
    print("Weights sample:", weights[0].tolist())

    # 测试 Top-2 路由
    y_top2, bal_loss_top2 = net(x, top_k=2)
    print("--- Top-2 Routing ---")
    print("Output shape: {}".format(y_top2.shape))

    print("NAFNetMOE test passed!")
