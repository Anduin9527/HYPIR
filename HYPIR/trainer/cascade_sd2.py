# -*- coding: utf-8 -*-
"""
阶段 2：级联 SD2 训练器
集成去模糊预处理的 HYPIR 训练器
- Blur 数据：先去模糊，再训练 HYPIR
- 其他数据：直接训练 HYPIR
- 去模糊模块冻结
"""
import torch
from accelerate.logging import get_logger

from HYPIR.trainer.sd2 import SD2Trainer
from HYPIR.model.isblur import BlurClassifier
from HYPIR.model.nafnet_wrapper import NAFNetWrapper


logger = get_logger(__name__, log_level="INFO")


class CascadeSD2Trainer(SD2Trainer):
    """集成去模糊预处理的 HYPIR 训练器
    
    在原有 SD2Trainer 基础上添加：
    1. 加载阶段1训练的去模糊模块（isBlur + NAFNet）
    2. 在 VAE 编码前对 LQ 应用去模糊预处理
    3. 去模糊模块冻结，不参与梯度更新
    """
    
    def __init__(self, config):
        # 先调用父类初始化（会调用 init_models）
        super().__init__(config)
        
        # 在 init_models 后加载去模糊模块
        self.init_deblur_module()
    
    def init_deblur_module(self):
        """加载阶段1训练的去模糊模块（冻结）"""
        if not hasattr(self.config, 'deblur_module_checkpoint'):
            # 如果没有指定去模糊模块，跳过
            self.use_deblur = False
            logger.warning("⚠️  未指定 deblur_module_checkpoint，跳过去模糊预处理")
            return
        
        checkpoint_path = self.config.deblur_module_checkpoint
        logger.info("=" * 60)
        logger.info(f"加载去模糊模块: {checkpoint_path}")
        
        self.use_deblur = True
        
        # PyTorch 2.6+ 默认 weights_only=True，但检查点包含 OmegaConf 配置对象
        # 由于这是自己训练的可信检查点，使用 weights_only=False 加载
        # map_location='cpu' 先加载到 CPU，然后再移动到指定设备，避免 GPU 冲突
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # 提取配置
        deblur_config = checkpoint.get('config', {})
        isblur_config = deblur_config.get('isblur', {})
        nafnet_config = deblur_config.get('nafnet', {})
        
        # 加载 isBlur
        logger.info("  - 加载 isBlur 分类器...")
        self.isblur = BlurClassifier(
            backbone=isblur_config.get('backbone', 'resnet18')
        )
        self.isblur.load_state_dict(checkpoint['isblur'])
        self.isblur.eval()
        for param in self.isblur.parameters():
            param.requires_grad = False
        
        # 加载 NAFNet
        logger.info("  - 加载 NAFNet 去模糊网络...")
        self.nafnet = NAFNetWrapper(
            checkpoint_path=None,  # 权重已经在 checkpoint 中
            width=nafnet_config.get('width', 64),
            enc_blks=nafnet_config.get('enc_blks', [2, 2, 4, 8]),
            middle_blk_num=nafnet_config.get('middle_blk_num', 12),
            dec_blks=nafnet_config.get('dec_blks', [2, 2, 2, 2]),
            freeze=True  # 冻结
        )
        self.nafnet.nafnet.load_state_dict(checkpoint['nafnet'])
        self.nafnet.eval()
        for param in self.nafnet.parameters():
            param.requires_grad = False
        
        # 移动到设备
        self.isblur = self.isblur.to(self.device)
        self.nafnet = self.nafnet.to(self.device)
        
        logger.info("✅ 去模糊模块加载完成（已冻结）")
        logger.info(f"  - 阈值: {self.config.get('blur_threshold', 0.5)}")
        logger.info("=" * 60)
    
    def preprocess_lq(self, lq):
        """
        对输入 LQ 应用去模糊预处理（如果启用）
        
        策略：使用 soft gating，根据 isBlur 概率混合去模糊结果和原图
        
        Args:
            lq: [B, 3, H, W], range [-1, 1]
        
        Returns:
            preprocessed_lq: [B, 3, H, W], range [-1, 1]
        """
        if not self.use_deblur:
            return lq
        
        with torch.no_grad():
            # isBlur 判断
            is_blur_logits = self.isblur(lq)
            is_blur_prob = torch.sigmoid(is_blur_logits)  # [B, 1]
            
            # NAFNet 去模糊
            deblurred = self.nafnet(lq)
            
            # 根据 isBlur 概率混合（soft gating）
            threshold = self.config.get('blur_threshold', 0.5)
            is_blur_mask = (is_blur_prob > threshold).float()
            
            # 混合：模糊图像用去模糊结果，非模糊图像保持原样
            # 扩展 mask 维度以匹配图像 [B, 1] -> [B, 1, 1, 1]
            is_blur_mask = is_blur_mask.view(-1, 1, 1, 1)
            output = is_blur_mask * deblurred + (1 - is_blur_mask) * lq
            
            return output
    
    def forward_generator(self):
        """重写：在 VAE 编码前应用去模糊预处理"""
        # 预处理 LQ（如果启用去模糊）
        lq_preprocessed = self.preprocess_lq(self.batch_inputs.lq)
        
        # VAE 编码（使用预处理后的 LQ）
        # 注意：z_lq 已在 prepare_batch_inputs 中创建，需要用 update 方法更新
        with torch.no_grad():
            z_lq = self.vae.encode(lq_preprocessed.to(self.weight_dtype)).latent_dist.sample()
        self.batch_inputs.update(z_lq=z_lq)
        
        # 后续流程与 SD2Trainer 相同
        z_in = self.batch_inputs.z_lq * self.vae.config.scaling_factor
        eps = self.G(
            z_in,
            self.batch_inputs.timesteps,
            encoder_hidden_states=self.batch_inputs.c_txt["text_embed"],
        ).sample
        z = self.scheduler.step(eps, self.config.coeff_t, z_in).pred_original_sample
        x = self.vae.decode(z.to(self.weight_dtype) / self.vae.config.scaling_factor).sample.float()
        
        return x


if __name__ == "__main__":
    # 测试导入
    print("CascadeSD2Trainer 导入成功！")
    print("该训练器继承自 SD2Trainer，添加了去模糊预处理功能。")
