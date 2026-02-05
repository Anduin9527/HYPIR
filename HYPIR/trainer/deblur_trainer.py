# -*- coding: utf-8 -*-
"""
阶段 1b：去模糊训练器
冻结 isBlur，微调 NAFNet 在 Blur 数据上
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from omegaconf import OmegaConf
from tqdm import tqdm
import pyiqa
from pathlib import Path
import swanlab

from HYPIR.model.isblur import BlurClassifier
from HYPIR.model.nafnet_wrapper import NAFNetWrapper
from HYPIR.dataset.blur_labeled import BlurLabeledDataset
from PIL import Image
import glob


logger = get_logger(__name__, log_level="INFO")


class DeblurTrainer:
    """阶段1b：微调 NAFNet（isBlur 冻结）
    
    目标：
    - 在 Blur 类型数据上微调 NAFNet
    - PSNR 提升 > 3dB
    - 输出包含 isBlur + NAFNet 的完整去模糊模块
    """
    
    def __init__(self, config):
        self.config = config
        
        # 初始化 Accelerator
        self.accelerator = Accelerator(
            mixed_precision=config.get('mixed_precision', 'bf16'),
            gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
            project_dir=config.output_dir,
        )
        
        # 创建输出目录
        if self.accelerator.is_main_process:
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 加载冻结的 isBlur
        logger.info("=" * 60)
        logger.info("加载 isBlur 分类器（冻结）...")
        self.isblur = BlurClassifier(backbone=config.isblur.backbone)
        state = torch.load(config.isblur.checkpoint, map_location='cpu')
        self.isblur.load_state_dict(state)
        self.isblur.eval()
        for param in self.isblur.parameters():
            param.requires_grad = False
        logger.info(f"✅ 从 {config.isblur.checkpoint} 加载 isBlur")
        
        # NAFNet（可训练）
        logger.info("=" * 60)
        logger.info("初始化 NAFNet（可训练）...")
        self.nafnet = NAFNetWrapper(
            checkpoint_path=config.nafnet.checkpoint,
            width=config.nafnet.get('width', 64),
            enc_blks=config.nafnet.get('enc_blks', [2, 2, 4, 8]),
            middle_blk_num=config.nafnet.get('middle_blk_num', 12),
            dec_blks=config.nafnet.get('dec_blks', [2, 2, 2, 2]),
            freeze=False  # 不冻结，要微调
        )
        trainable_params = sum(p.numel() for p in self.nafnet.parameters() if p.requires_grad) / 1e6
        logger.info(f"可训练参数: {trainable_params:.2f}M")
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.nafnet.parameters(),
            lr=config.learning_rate,
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # 数据集（只用 Blur 类型）
        logger.info("=" * 60)
        logger.info("加载数据集...")
        dataset = BlurLabeledDataset(
            file_meta={'file_list': config.data.file_list},
            out_size=config.data.image_size,
            crop_type='none',
            use_hflip=True,
            use_rot=True,
        )
        
        # 过滤只保留 Blur 样本
        original_size = len(dataset)
        dataset.data_rows = [row for row in dataset.data_rows if 'Blur' in row['lq_path']]
        logger.info(f"Blur 样本数: {len(dataset)} / {original_size}")
        
        if len(dataset) == 0:
            raise ValueError("没有找到 Blur 样本！请检查数据集路径是否包含 'Blur' 关键词")
        
        self.train_loader = DataLoader(
            dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            pin_memory=True,
        )
        
        # Metrics
        self.metric_psnr = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr')
        self.metric_ssim = pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr')
        
        # 验证集（只用 Blur 类型）
        self.val_samples = []
        if hasattr(config, 'val'):
            logger.info("=" * 60)
            logger.info("加载验证集...")
            val_dir = Path(config.val.val_dir)
            for deg_type in config.val.degradation_types:
                lq_dir = val_dir / deg_type / 'LQ'
                gt_dir = val_dir / deg_type / 'GT'
                
                if not lq_dir.exists() or not gt_dir.exists():
                    continue
                
                lq_files = sorted(glob.glob(str(lq_dir / '*.png')) + glob.glob(str(lq_dir / '*.jpg')))
                num_samples = config.val.get('num_samples_per_type')
                if num_samples is not None:
                    lq_files = lq_files[:num_samples]
                
                for lq_path in lq_files:
                    filename = Path(lq_path).name
                    gt_path = gt_dir / filename
                    if gt_path.exists():
                        self.val_samples.append({
                            'lq_path': lq_path,
                            'gt_path': str(gt_path),
                            'degradation_type': deg_type
                        })
            
            logger.info(f"验证样本数: {len(self.val_samples)}")
        
        # Prepare
        self.isblur, self.nafnet, self.optimizer, self.train_loader = self.accelerator.prepare(
            self.isblur, self.nafnet, self.optimizer, self.train_loader
        )
        
        # 初始化 SwanLab (仅在主进程)
        self.swanlab_run = None
        if self.accelerator.is_main_process:
            logger.info("=" * 60)
            logger.info("初始化 SwanLab 监控...")
            try:
                self.swanlab_run = swanlab.init(
                    project="HYPIR",
                    experiment_name=f"deblur-{config.get('experiment_name', 'nafnet-finetune')}",
                    config=OmegaConf.to_container(config, resolve=True),
                    description="Stage 1b: NAFNet Deblur Module Fine-tuning"
                )
                logger.info("✅ SwanLab 初始化成功")
            except Exception as e:
                logger.warning(f"⚠️ SwanLab 初始化失败: {e}")
                logger.warning("训练将继续，但不会记录到 SwanLab")
        
        logger.info("=" * 60)
        logger.info("初始化完成！")
    
    def train(self):
        """训练循环"""
        best_psnr = 0.0
        global_step = 0
        
        logger.info("=" * 60)
        logger.info("开始训练...")
        logger.info("=" * 60)
        
        for epoch in range(self.config.num_epochs):
            self.nafnet.train()
            total_loss = 0
            total_l1 = 0
            total_l2 = 0
            
            pbar = tqdm(
                self.train_loader, 
                disable=not self.accelerator.is_local_main_process,
                desc=f"Epoch {epoch+1}/{self.config.num_epochs}"
            )
            
            for batch_idx, batch in enumerate(pbar):
                lq = batch['LQ'] * 2 - 1  # [0, 1] -> [-1, 1]
                gt = batch['GT'] * 2 - 1
                
                # isBlur 判断（冻结，仅用于统计）
                with torch.no_grad():
                    is_blur_logits = self.isblur(lq)
                    is_blur_prob = torch.sigmoid(is_blur_logits).mean().item()
                
                # NAFNet 去模糊
                deblurred = self.nafnet(lq)
                
                # Loss：L1 + L2 混合
                loss_l1 = F.l1_loss(deblurred, gt)
                loss_l2 = F.mse_loss(deblurred, gt)
                loss = self.config.get('lambda_l1', 1.0) * loss_l1 + \
                       self.config.get('lambda_l2', 1.0) * loss_l2
                
                # Backward
                self.accelerator.backward(loss)
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                
                total_loss += loss.item()
                total_l1 += loss_l1.item()
                total_l2 += loss_l2.item()
                
                # 记录到 SwanLab
                if self.swanlab_run is not None and (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    swanlab.log({
                        'loss/total': loss.item(),
                        'loss/l1': loss_l1.item(),
                        'loss/l2': loss_l2.item(),
                        'train/blur_prob': is_blur_prob,
                        'train/learning_rate': self.optimizer.param_groups[0]['lr']
                    }, step=global_step)
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'l1': f'{loss_l1.item():.4f}',
                    'l2': f'{loss_l2.item():.4f}',
                    'blur_prob': f'{is_blur_prob:.2f}'
                })
            
            # Validation
            val_psnr, val_ssim = self.validate()
            
            # Epoch 统计
            avg_loss = total_loss / len(self.train_loader)
            avg_l1 = total_l1 / len(self.train_loader)
            avg_l2 = total_l2 / len(self.train_loader)
            
            # 记录 epoch 级别的指标到 SwanLab
            if self.swanlab_run is not None:
                swanlab.log({
                    'epoch': epoch + 1,
                    'loss/epoch_avg': avg_loss,
                    'loss/epoch_l1': avg_l1,
                    'loss/epoch_l2': avg_l2,
                    'val/psnr': val_psnr,
                    'val/ssim': val_ssim
                }, step=global_step)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} 完成")
            logger.info(f"  Loss: {avg_loss:.4f} (L1: {avg_l1:.4f}, L2: {avg_l2:.4f})")
            logger.info(f"  Val PSNR: {val_psnr:.2f} dB")
            logger.info(f"  Val SSIM: {val_ssim:.4f}")
            
            # 保存最佳模型
            if val_psnr > best_psnr and self.accelerator.is_main_process:
                best_psnr = val_psnr
                self.save_checkpoint('deblur_module_best.pth')
                logger.info(f"✅ 保存最佳模型 (PSNR: {best_psnr:.2f})")
            
            # 每个 epoch 保存
            if self.accelerator.is_main_process:
                self.save_checkpoint(f'deblur_module_epoch_{epoch+1}.pth')
        
        # 训练完成
        logger.info("=" * 60)
        logger.info("✅ 训练完成！")
        logger.info(f"最佳 PSNR: {best_psnr:.2f} dB")
        logger.info(f"模型保存在: {self.config.output_dir}")
        logger.info("=" * 60)
        
        # 关闭 SwanLab
        if self.swanlab_run is not None:
            swanlab.finish()
            logger.info("✅ SwanLab 记录已保存")
    
    def validate(self):
        """验证（使用独立验证集）"""
        if not self.val_samples:
            # 如果没有验证集，回退到训练集验证
            return self._validate_on_train()
        
        self.nafnet.eval()
        val_psnr = 0
        val_ssim = 0
        val_count = 0
        
        from torchvision import transforms
        to_tensor = transforms.ToTensor()
        
        with torch.no_grad():
            for sample in tqdm(self.val_samples, desc="Validating", disable=not self.accelerator.is_local_main_process):
                # 加载图像
                lq_img = Image.open(sample['lq_path']).convert('RGB')
                gt_img = Image.open(sample['gt_path']).convert('RGB')
                
                # 调整大小
                target_size = self.config.data.image_size
                if lq_img.size != (target_size, target_size):
                    lq_img = lq_img.resize((target_size, target_size), Image.BICUBIC)
                    gt_img = gt_img.resize((target_size, target_size), Image.BICUBIC)
                
                # 转换为 tensor
                lq = to_tensor(lq_img).unsqueeze(0) * 2 - 1  # [0,1] -> [-1,1]
                gt = to_tensor(gt_img).unsqueeze(0) * 2 - 1
                
                lq = lq.to(self.accelerator.device)
                gt = gt.to(self.accelerator.device)
                
                # NAFNet 去模糊
                deblurred = self.nafnet(lq)
                
                # 转换到 [0, 1] for metrics，并 clamp 确保在有效范围内
                deblurred_01 = torch.clamp((deblurred + 1) / 2, 0, 1)
                gt_01 = torch.clamp((gt + 1) / 2, 0, 1)
                
                # 计算指标
                psnr = self.metric_psnr(deblurred_01, gt_01).mean().item()
                ssim = self.metric_ssim(deblurred_01, gt_01).mean().item()
                
                val_psnr += psnr
                val_ssim += ssim
                val_count += 1
        
        self.nafnet.train()
        
        return val_psnr / val_count, val_ssim / val_count
    
    def _validate_on_train(self):
        """备用：在训练集上验证（前 N 个 batch）"""
        self.nafnet.eval()
        val_psnr = 0
        val_ssim = 0
        val_count = 0
        
        with torch.no_grad():
            val_batches = min(10, len(self.train_loader))
            for i, batch in enumerate(self.train_loader):
                if i >= val_batches:
                    break
                
                lq = batch['LQ'] * 2 - 1
                gt = batch['GT'] * 2 - 1
                deblurred = self.nafnet(lq)
                
                # Clamp 确保在 [0, 1] 范围内
                deblurred_01 = torch.clamp((deblurred + 1) / 2, 0, 1)
                gt_01 = torch.clamp((gt + 1) / 2, 0, 1)
                
                psnr = self.metric_psnr(deblurred_01, gt_01).mean().item()
                ssim = self.metric_ssim(deblurred_01, gt_01).mean().item()
                
                val_psnr += psnr
                val_ssim += ssim
                val_count += 1
        
        self.nafnet.train()
        
        return val_psnr / val_count, val_ssim / val_count
    
    def save_checkpoint(self, filename):
        """保存完整的去模糊模块（isBlur + NAFNet）"""
        save_path = Path(self.config.output_dir) / filename
        
        unwrapped_isblur = self.accelerator.unwrap_model(self.isblur)
        unwrapped_nafnet = self.accelerator.unwrap_model(self.nafnet)
        
        checkpoint = {
            'isblur': unwrapped_isblur.state_dict(),
            'nafnet': unwrapped_nafnet.nafnet.state_dict(),  # 保存内部的 NAFNet
            'config': dict(self.config)
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"保存 checkpoint: {save_path}")


if __name__ == "__main__":
    # 简单测试
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    config = OmegaConf.load(args.config)
    trainer = DeblurTrainer(config)
    trainer.train()
