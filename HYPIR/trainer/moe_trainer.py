# -*- coding: utf-8 -*-
"""
MOE 训练器
用于训练 NAFNet+MOE 的轻量级 Router，并支持负载均衡损失
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
import glob
from PIL import Image

from HYPIR.model.nafnet_moe import NAFNetMOE
from HYPIR.dataset.blur_labeled import BlurLabeledDataset

logger = get_logger(__name__, log_level="INFO")


class MoeTrainer:
    """NAFNet + MOE 训练器

    目标：
    - 读取 5 个预训练的 NAFNet 权重并冻结 (或者设定极小的学习率)
    - 训练 Router 网络分配权重
    """

    def __init__(self, config):
        self.config = config

        # 初始化 Accelerator
        self.accelerator = Accelerator(
            mixed_precision=config.get("mixed_precision", "bf16"),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
            project_dir=config.output_dir,
        )

        if self.accelerator.is_main_process:
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("初始化 NAFNet+MOE 架构...")

        # 初始化 NAFNetMOE 网络
        self.moe_net = NAFNetMOE(
            num_experts=config.moe.get("num_experts", 5),
            router_base_channels=config.moe.get("router_channels", 32),
            img_channel=3,
            width=config.nafnet.get("width", 32),
            middle_blk_num=config.nafnet.get("middle_blk_num", 1),
            enc_blk_nums=config.nafnet.get("enc_blk_nums", [1, 1, 1, 28]),
            dec_blk_nums=config.nafnet.get("dec_blk_nums", [1, 1, 1, 1]),
        )

        # 加载专家权重 (如果有配置)
        expert_ckpts = config.moe.get("expert_checkpoints", [])
        if expert_ckpts and len(expert_ckpts) == self.moe_net.num_experts:
            logger.info("正在加载 5 个专家网络的预训练权重...")
            for i, ckpt_path in enumerate(expert_ckpts):
                if ckpt_path and Path(ckpt_path).exists():
                    state_dict = torch.load(ckpt_path, map_location="cpu")
                    # 取决于之前的保存格式，可能需要 .get('nafnet', state_dict)
                    if "nafnet" in state_dict:
                        state_dict = state_dict["nafnet"]
                    self.moe_net.experts[i].load_state_dict(state_dict)
                    logger.info(f"✅ Expert {i} weights loaded from {ckpt_path}")
                else:
                    logger.warning(
                        f"⚠️ Expert {i} weights null or not found at {ckpt_path}"
                    )

        # 冻结专家参数 (推荐)，专门训练 Router
        if config.moe.get("freeze_experts", True):
            self.moe_net.freeze_experts()
            logger.info("✅ 已冻结所有专家 NAFNet，仅训练 Router")
            trainable_params = (
                sum(
                    p.numel()
                    for p in self.moe_net.router.parameters()
                    if p.requires_grad
                )
                / 1e6
            )
        else:
            self.moe_net.unfreeze_experts()
            logger.info("⚠️ 联合微调：Router 和 Experts 的参数都在更新！")
            trainable_params = (
                sum(p.numel() for p in self.moe_net.parameters() if p.requires_grad)
                / 1e6
            )

        logger.info(f"可训练参数: {trainable_params:.2f}M")

        # 优化器
        # 这里默认把所有 requires_grad=True 的参数传进去
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.moe_net.parameters()),
            lr=config.learning_rate,
            weight_decay=config.get("weight_decay", 1e-4),
        )

        # 数据集 (采用与原版 trainer 类似的混合降质的数据集，建议不要做过滤了)
        logger.info("=" * 60)
        logger.info("加载数据集...")
        # 注意：这里的 Dataset 可能不仅是 Blur，可能还需要 Noise/JPEG
        dataset = BlurLabeledDataset(
            file_meta={"file_list": config.data.file_list},
            out_size=config.data.image_size,
            crop_type=config.data.get("crop_type", "none"),
            use_hflip=True,
            use_rot=True,
        )

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

        # Prepare Accelerator
        self.moe_net, self.optimizer, self.train_loader = self.accelerator.prepare(
            self.moe_net, self.optimizer, self.train_loader
        )

        # SwanLab
        self.swanlab_run = None
        if self.accelerator.is_main_process:
            logger.info("=" * 60)
            logger.info("初始化 SwanLab 监控...")
            try:
                self.swanlab_run = swanlab.init(
                    project="HYPIR",
                    experiment_name=f"moe-{config.get('experiment_name', 'router-train')}",
                    config=OmegaConf.to_container(config, resolve=True),
                    description="Training the Router for NAFNet+MOE Architecture",
                )
                logger.info("✅ SwanLab 初始化成功")
            except Exception as e:
                logger.warning(f"⚠️ SwanLab 初始化失败: {e}")

        logger.info("初始化完成！")

    def train(self):
        best_psnr = 0.0
        global_step = 0
        lambda_bal = self.config.moe.get("lambda_balance", 0.1)  # 负载均衡惩罚权重

        logger.info("开始训练 MOE 架构...")

        for epoch in range(self.config.num_epochs):
            self.moe_net.train()
            # 由于可能冻结了 expert，专家实际上处于什么 status 不影响梯度

            total_loss = 0
            total_l1 = 0
            total_l2 = 0
            total_bal = 0

            pbar = tqdm(
                self.train_loader,
                disable=not self.accelerator.is_local_main_process,
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
            )

            for batch_idx, batch in enumerate(pbar):
                lq = batch["LQ"] * 2 - 1  # [-1, 1]
                gt = batch["GT"] * 2 - 1  # [-1, 1]

                # NAFNet MOE 前向，返回修复结果和负载均衡 Loss
                # 注意：训练时 top_k=None 开启全量软路由
                restored, balance_loss, routing_weights = self.moe_net(
                    lq, top_k=None, return_routing=True
                )

                # 重建 Loss
                loss_l1 = F.l1_loss(restored, gt)
                loss_l2 = F.mse_loss(restored, gt)

                loss_recon = (
                    self.config.get("lambda_l1", 1.0) * loss_l1
                    + self.config.get("lambda_l2", 1.0) * loss_l2
                )

                # 总 Loss = 重建 + 负载均衡
                loss = loss_recon + lambda_bal * balance_loss

                self.accelerator.backward(loss)

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # 获取梯度的参数组 (主要是 router)
                    # self.accelerator.clip_grad_norm_(self.moe_net.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                total_loss += loss.item()
                total_l1 += loss_l1.item()
                total_l2 += loss_l2.item()
                total_bal += balance_loss.item()

                if (
                    self.swanlab_run is not None
                    and (batch_idx + 1) % self.config.gradient_accumulation_steps == 0
                ):
                    swan_dict = {
                        "loss/total": loss.item(),
                        "loss/recon": loss_recon.item(),
                        "loss/balance": balance_loss.item(),
                        "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                    }
                    # 记录专家的平均被选概率
                    mean_w = routing_weights.mean(dim=0)
                    for i in range(mean_w.size(0)):
                        swan_dict[f"router/expert_{i}_prob"] = mean_w[i].item()

                    swanlab.log(swan_dict, step=global_step)

                pbar.set_postfix(
                    {
                        "L": f"{loss.item():.3f}",
                        "Rec": f"{loss_recon.item():.3f}",
                        "Bal": f"{balance_loss.item():.3f}",
                    }
                )

            # Epoch 结算与验证
            # 暂借训练集当简单的 validation 测试
            val_psnr, val_ssim = self._validate_on_train()
            avg_loss = total_loss / len(self.train_loader)
            avg_bal = total_bal / len(self.train_loader)

            if self.swanlab_run is not None:
                swanlab.log(
                    {
                        "epoch": epoch + 1,
                        "loss/epoch_avg": avg_loss,
                        "loss/epoch_bal": avg_bal,
                        "val/psnr": val_psnr,
                        "val/ssim": val_ssim,
                    },
                    step=global_step,
                )

            logger.info(
                f"\nEpoch {epoch + 1} 完成 - Loss: {avg_loss:.4f} (Bal: {avg_bal:.4f}), Val PSNR: {val_psnr:.2f} dB"
            )

            if val_psnr > best_psnr and self.accelerator.is_main_process:
                best_psnr = val_psnr
                self.save_checkpoint("moe_best.pth")

            if (
                self.accelerator.is_main_process
                and (epoch + 1) % self.config.get("save_freq", 5) == 0
            ):
                self.save_checkpoint(f"moe_epoch_{epoch + 1}.pth")

        if self.swanlab_run is not None:
            swanlab.finish()

    def _validate_on_train(self):
        """简单验证（使用评估期）"""
        self.moe_net.eval()
        val_psnr, val_ssim, val_count = 0, 0, 0

        with torch.no_grad():
            val_batches = min(10, len(self.train_loader))
            for i, batch in enumerate(self.train_loader):
                if i >= val_batches:
                    break

                lq = batch["LQ"] * 2 - 1
                gt = batch["GT"] * 2 - 1

                # 验证时可以测试开启 top_k=2 (软硬结合) 或直接全开
                restored, _ = self.moe_net(lq, top_k=2)

                res_01 = torch.clamp((restored + 1) / 2, 0, 1)
                gt_01 = torch.clamp((gt + 1) / 2, 0, 1)

                val_psnr += self.metric_psnr(res_01, gt_01).mean().item()
                val_ssim += self.metric_ssim(res_01, gt_01).mean().item()
                val_count += 1

        self.moe_net.train()
        return val_psnr / val_count, val_ssim / val_count

    def save_checkpoint(self, filename):
        save_path = Path(self.config.output_dir) / filename
        unwrapped = self.accelerator.unwrap_model(self.moe_net)

        torch.save(
            {"moe_net": unwrapped.state_dict(), "config": dict(self.config)}, save_path
        )
        logger.info(f"保存 checkpoint: {save_path}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    trainer = MoeTrainer(config)
    trainer.train()
