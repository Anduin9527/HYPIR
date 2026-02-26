# -*- coding: utf-8 -*-
"""
阶段 1a：预先训练单个 NAFNet Expert
只针对单一退化类型进行微调 (如 Blur, Rain, Noise 等)
不带有 ISBLUR 判断
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

# 这里使用只包装了一个本地 NAFNet 的 NAFNetWrapper
from HYPIR.model.nafnet_wrapper import NAFNetWrapper
from HYPIR.dataset.expert import ExpertDataset

logger = get_logger(__name__, log_level="INFO")


class ExpertTrainer:
    """阶段1a：预热并微调基础 NAFNet (Expert)"""

    def __init__(self, config):
        self.config = config
        self.expert_type = config.get("expert_type", "Blur")

        # 初始化 Accelerator
        self.accelerator = Accelerator(
            mixed_precision=config.get("mixed_precision", "bf16"),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
            project_dir=config.output_dir,
        )

        # 创建输出目录
        if self.accelerator.is_main_process:
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info(f"初始化 {self.expert_type} 专家 NAFNet...")
        self.nafnet = NAFNetWrapper(
            checkpoint_path=config.nafnet.get("checkpoint", None),
            width=config.nafnet.get("width", 64),
            enc_blks=config.nafnet.get("enc_blks", [2, 2, 4, 8]),
            middle_blk_num=config.nafnet.get("middle_blk_num", 12),
            dec_blks=config.nafnet.get("dec_blks", [2, 2, 2, 2]),
            freeze=False,  # 完全开放训练
        )
        trainable_params = (
            sum(p.numel() for p in self.nafnet.parameters() if p.requires_grad) / 1e6
        )
        logger.info(f"可训练参数: {trainable_params:.2f}M")

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.nafnet.parameters(),
            lr=config.learning_rate,
            weight_decay=config.get("weight_decay", 1e-4),
        )

        # 数据集（按 expert_type 过滤）
        logger.info("=" * 60)
        logger.info(f"加载 {self.expert_type} 数据集...")
        try:
            dataset = ExpertDataset(
                file_meta={"file_list": config.data.file_list},
                out_size=config.data.image_size,
                expert_type=self.expert_type,
                crop_type="none",
                use_hflip=True,
                use_rot=True,
            )
        except Exception as e:
            logger.error(
                f"加载数据集失败，请检查 parquet 文件路径 {config.data.file_list}，报错: {e}"
            )
            raise e

        logger.info(
            f"{self.expert_type} 样本数过滤后: {len(dataset)} / 原始: {getattr(dataset, 'original_size', 'N/A')}"
        )

        if len(dataset) == 0:
            raise ValueError(f"没有找到 {self.expert_type} 的样本！")

        self.train_loader = DataLoader(
            dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            pin_memory=True,
        )

        # Metrics
        self.metric_psnr = pyiqa.create_metric(
            "psnr", test_y_channel=True, color_space="ycbcr"
        )
        self.metric_ssim = pyiqa.create_metric(
            "ssim", test_y_channel=True, color_space="ycbcr"
        )
        # Suppress warnings from lpips since it uses torchvision
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # import lpips as the original repo does, or load pyiqa lpips
            self.metric_lpips = pyiqa.create_metric("lpips", net="vgg").to(
                self.accelerator.device
            )
            self.metric_lpips.eval()
            self.metric_lpips.requires_grad_(False)

        # Prepare
        self.nafnet, self.optimizer, self.train_loader = self.accelerator.prepare(
            self.nafnet, self.optimizer, self.train_loader
        )

        # 初始化 SwanLab (仅在主进程)
        self.swanlab_run = None
        if self.accelerator.is_main_process:
            logger.info("初始化 SwanLab 监控...")
            try:
                self.swanlab_run = swanlab.init(
                    project="HYPIR",
                    experiment_name=f"expert-{self.expert_type}-{config.get('experiment_name', 'nafnet-train')}",
                    config=OmegaConf.to_container(config, resolve=True),
                    description=f"Pretraining {self.expert_type} NAFNet Expert Module",
                )
            except Exception as e:
                logger.warning(f"⚠️ SwanLab 初始化失败: {e}")

    def train(self):
        best_score = -float("inf")  # Use joint Score instead of just PSNR
        global_step = 0

        logger.info("=" * 60)
        logger.info(f"开始 {self.expert_type} 专家预备训练...")

        for epoch in range(self.config.num_epochs):
            self.nafnet.train()
            total_loss, total_l1, total_l2 = 0, 0, 0

            pbar = tqdm(
                self.train_loader,
                disable=not self.accelerator.is_local_main_process,
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
            )

            for batch_idx, batch in enumerate(pbar):
                lq = batch["LQ"] * 2 - 1  # [0, 1] -> [-1, 1]
                gt = batch["GT"] * 2 - 1

                deblurred = self.nafnet(lq)
                loss_l1 = F.l1_loss(deblurred, gt)
                loss_l2 = F.mse_loss(deblurred, gt)
                loss = (
                    self.config.get("lambda_l1", 1.0) * loss_l1
                    + self.config.get("lambda_l2", 1.0) * loss_l2
                )

                self.accelerator.backward(loss)

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                total_loss += loss.item()
                total_l1 += loss_l1.item()
                total_l2 += loss_l2.item()

                if (
                    self.swanlab_run is not None
                    and (batch_idx + 1) % self.config.gradient_accumulation_steps == 0
                ):
                    swanlab.log(
                        {
                            "loss/total": loss.item(),
                            "loss/l1": loss_l1.item(),
                            "loss/l2": loss_l2.item(),
                            "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                        },
                        step=global_step,
                    )

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # 简单的验证（目前从训练集中取10批次简单算一下PSNR）
            val_psnr, val_ssim, val_lpips, val_score = self._validate_on_train()

            avg_loss = total_loss / len(self.train_loader)
            if self.swanlab_run is not None:
                swanlab.log(
                    {
                        "epoch": epoch + 1,
                        "val/psnr": val_psnr,
                        "val/ssim": val_ssim,
                        "val/lpips": val_lpips,
                        "val/score": val_score,
                    },
                    step=global_step,
                )

            logger.info(
                f"\n{self.expert_type} Epoch {epoch + 1} - Loss: {avg_loss:.4f} PSNR: {val_psnr:.2f} dB, Score: {val_score:.2f}"
            )

            if val_score > best_score and self.accelerator.is_main_process:
                best_score = val_score
                self.save_checkpoint(f"{self.expert_type}_expert_best.pth")

            if (
                self.accelerator.is_main_process
                and (epoch + 1) % self.config.get("save_freq", 5) == 0
            ):
                self.save_checkpoint(f"{self.expert_type}_expert_epoch_{epoch + 1}.pth")

        if self.swanlab_run is not None:
            swanlab.finish()

    def _validate_on_train(self):
        self.nafnet.eval()
        val_psnr, val_ssim, val_lpips, val_count = 0, 0, 0, 0

        with torch.no_grad():
            val_batches = min(10, len(self.train_loader))
            for i, batch in enumerate(self.train_loader):
                if i >= val_batches:
                    break

                lq = batch["LQ"] * 2 - 1
                gt = batch["GT"] * 2 - 1
                deblurred = self.nafnet(lq)

                res_01 = torch.clamp((deblurred + 1) / 2, 0, 1)
                gt_01 = torch.clamp((gt + 1) / 2, 0, 1)

                val_psnr += self.metric_psnr(res_01, gt_01).mean().item()
                val_ssim += self.metric_ssim(res_01, gt_01).mean().item()

                # 計算 LPIPS，注意 pyiqa lpips 預期輸入也是 [0, 1] 範圍
                val_lpips += self.metric_lpips(res_01, gt_01).mean().item()
                val_count += 1

        self.nafnet.train()
        avg_psnr = val_psnr / val_count
        avg_ssim = val_ssim / val_count
        avg_lpips = val_lpips / val_count

        # 綜合 Score 公式: PSNR + 10 * SSIM - 5 * LPIPS
        score = avg_psnr + 10.0 * avg_ssim - 5.0 * avg_lpips

        return avg_psnr, avg_ssim, avg_lpips, score

    def save_checkpoint(self, filename):
        save_path = Path(self.config.output_dir) / filename
        unwrapped_nafnet = self.accelerator.unwrap_model(self.nafnet)

        checkpoint = {
            "nafnet": unwrapped_nafnet.nafnet.state_dict(),
            "config": dict(self.config),
            "expert_type": self.expert_type,
        }

        torch.save(checkpoint, save_path)
        logger.info(f"保存 Expert {self.expert_type} checkpoint: {save_path}")
