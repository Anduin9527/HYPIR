# -*- coding: utf-8 -*-
"""
NAFNet: Simple Baselines for Image Restoration
迁移自 NAFNet 官方实现，移除外部依赖

Citation:
@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# LayerNorm2d 实现（从 basicsr 迁移）
# ============================================================================

class LayerNormFunction(torch.autograd.Function):
    """自定义 LayerNorm 的前向和反向传播"""
    
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )


class LayerNorm2d(nn.Module):
    """2D LayerNorm for NAFNet"""
    
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


# ============================================================================
# NAFNet 核心组件
# ============================================================================

class SimpleGate(nn.Module):
    """Simple Gate 激活函数"""
    
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """NAFNet 基本模块"""
    
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        
        # Depthwise Convolution Branch
        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, stride=1, 
                               groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, kernel_size=1, padding=0, stride=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        # Feed-Forward Network
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, padding=0, stride=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, padding=0, stride=1, bias=True)

        # Normalization
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        # Dropout
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        # Learnable scaling parameters
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        # Depthwise Convolution Branch
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)

        y = inp + x * self.beta

        # Feed-Forward Network
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma


# ============================================================================
# NAFNet 主网络
# ============================================================================

class NAFNet(nn.Module):
    """NAFNet for Image Restoration"""
    
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        # Input/Output layers
        self.intro = nn.Conv2d(img_channel, width, kernel_size=3, padding=1, stride=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, kernel_size=3, padding=1, stride=1, bias=True)

        # Encoder/Decoder/Middle blocks
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        # Build Encoder
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )
            self.downs.append(nn.Conv2d(chan, 2*chan, 2, 2))
            chan = chan * 2

        # Build Middle blocks
        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        # Build Decoder
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        # Encoder
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        # Middle
        x = self.middle_blks(x)

        # Decoder
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        # Output
        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        """Pad input to be divisible by padder_size"""
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == '__main__':
    # 测试 NAFNet
    print("Testing NAFNet...")
    
    img_channel = 3
    width = 64
    enc_blks = [2, 2, 4, 8]
    middle_blk_num = 12
    dec_blks = [2, 2, 2, 2]
    
    net = NAFNet(
        img_channel=img_channel, 
        width=width, 
        middle_blk_num=middle_blk_num,
        enc_blk_nums=enc_blks, 
        dec_blk_nums=dec_blks
    )
    
    # 测试前向传播
    x = torch.randn(2, 3, 256, 256)
    y = net(x)
    
    print("Input shape: {}".format(x.shape))
    print("Output shape: {}".format(y.shape))
    
    # Calculate parameters
    params = sum(p.numel() for p in net.parameters()) / 1e6
    print("Parameters: {:.2f}M".format(params))
    
    print("NAFNet test passed!")
