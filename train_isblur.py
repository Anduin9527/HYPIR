# -*- coding: utf-8 -*-
"""
阶段 1a：训练 isBlur 分类器
识别输入图像是否为模糊图像的二分类任务
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from omegaconf import OmegaConf
from tqdm import tqdm
from argparse import ArgumentParser
import os
from pathlib import Path

from HYPIR.model.isblur import BlurClassifier
from HYPIR.dataset.blur_labeled import BlurLabeledDataset
from PIL import Image
import glob


def create_val_dataset(val_config, image_size):
    """创建验证数据集（从 Val 目录加载）"""
    val_dir = val_config['val_dir']
    degradation_types = val_config['degradation_types']
    num_samples = val_config.get('num_samples_per_type')
    
    val_samples = []
    for deg_type in degradation_types:
        lq_dir = Path(val_dir) / deg_type / 'LQ'
        if not lq_dir.exists():
            continue
        
        lq_files = sorted(glob.glob(str(lq_dir / '*.png')) + glob.glob(str(lq_dir / '*.jpg')))
        
        if num_samples is not None:
            lq_files = lq_files[:num_samples]
        
        for lq_path in lq_files:
            val_samples.append({
                'lq_path': lq_path,
                'is_blur': 1.0 if deg_type == 'Blur' else 0.0,
                'degradation_type': deg_type
            })
    
    return val_samples


def main():
    parser = ArgumentParser(description="训练 isBlur 分类器")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    
    # 初始化 Accelerator
    accelerator = Accelerator(
        mixed_precision=config.get('mixed_precision', 'bf16'),
        project_dir=config.output_dir,
    )
    
    # 创建输出目录
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        accelerator.print(f"输出目录: {config.output_dir}")
    
    # 创建模型
    accelerator.print("=" * 60)
    accelerator.print("初始化 isBlur 分类器...")
    model = BlurClassifier(
        backbone=config.get('backbone', 'resnet18'),
        pretrained=True
    )
    accelerator.print(f"使用 backbone: {config.get('backbone', 'resnet18')}")
    accelerator.print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 创建数据集
    accelerator.print("=" * 60)
    accelerator.print("加载数据集...")
    train_dataset = BlurLabeledDataset(
        file_meta={'file_list': config.data.file_list},
        out_size=config.data.image_size,
        crop_type='none',
        use_hflip=True,
        use_rot=False,
    )
    accelerator.print(f"训练样本数: {len(train_dataset)}")
    
    # 统计 Blur/非Blur 样本数
    blur_count = sum(1 for i in range(len(train_dataset)) 
                     if 'Blur' in train_dataset.data_rows[i]['lq_path'])
    accelerator.print(f"  - Blur 样本: {blur_count}")
    accelerator.print(f"  - 非Blur 样本: {len(train_dataset) - blur_count}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )
    
    # 创建验证集
    val_samples = None
    if hasattr(config, 'val'):
        accelerator.print("=" * 60)
        accelerator.print("加载验证集...")
        val_samples = create_val_dataset(config.val, config.data.image_size)
        accelerator.print(f"验证样本数: {len(val_samples)}")
        val_blur = sum(1 for s in val_samples if s['is_blur'] == 1.0)
        accelerator.print(f"  - Blur 样本: {val_blur}")
        accelerator.print(f"  - 非Blur 样本: {len(val_samples) - val_blur}")
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    # Loss
    criterion = nn.BCEWithLogitsLoss()
    
    # Accelerate 准备
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    
    # 训练循环
    accelerator.print("=" * 60)
    accelerator.print("开始训练...")
    accelerator.print("=" * 60)
    
    best_acc = 0.0
    global_step = 0
    
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # 统计每个类别的准确率
        blur_correct = 0
        blur_total = 0
        non_blur_correct = 0
        non_blur_total = 0
        
        pbar = tqdm(
            train_loader, 
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch+1}/{config.num_epochs}"
        )
        
        for batch in pbar:
            lq = batch['LQ'] * 2 - 1  # [0, 1] -> [-1, 1]
            labels = batch['is_blur'].unsqueeze(1).float()
            
            # Forward
            logits = model(lq)
            loss = criterion(logits, labels)
            
            # Backward
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            # Metrics
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct_batch = (preds == labels).sum().item()
            batch_size = labels.size(0)
            
            correct += correct_batch
            total += batch_size
            total_loss += loss.item()
            
            # 分类统计
            for i in range(batch_size):
                if labels[i].item() == 1.0:  # Blur
                    blur_total += 1
                    if preds[i].item() == labels[i].item():
                        blur_correct += 1
                else:  # 非Blur
                    non_blur_total += 1
                    if preds[i].item() == labels[i].item():
                        non_blur_correct += 1
            
            # 更新进度条
            current_acc = correct / total
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.4f}'
            })
            
            global_step += 1
        
        # Epoch 统计
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        blur_acc = blur_correct / blur_total if blur_total > 0 else 0
        non_blur_acc = non_blur_correct / non_blur_total if non_blur_total > 0 else 0
        
        accelerator.print(f"\n{'='*60}")
        accelerator.print(f"Epoch {epoch+1}/{config.num_epochs} 完成")
        accelerator.print(f"  Train Loss: {avg_loss:.4f}")
        accelerator.print(f"  Train Accuracy: {accuracy:.4f}")
        accelerator.print(f"  Blur Accuracy: {blur_acc:.4f} ({blur_correct}/{blur_total})")
        accelerator.print(f"  Non-Blur Accuracy: {non_blur_acc:.4f} ({non_blur_correct}/{non_blur_total})")
        
        # 验证集评估
        val_acc = accuracy  # 默认使用训练准确率
        if val_samples is not None:
            model.eval()
            val_correct = 0
            val_blur_correct = 0
            val_blur_total = 0
            val_non_blur_correct = 0
            val_non_blur_total = 0
            
            from torchvision import transforms
            to_tensor = transforms.ToTensor()
            
            with torch.no_grad():
                for sample in val_samples:
                    # 加载图像
                    img = Image.open(sample['lq_path']).convert('RGB')
                    if img.size != (config.data.image_size, config.data.image_size):
                        img = img.resize((config.data.image_size, config.data.image_size))
                    
                    img_tensor = to_tensor(img).unsqueeze(0) * 2 - 1  # [0,1] -> [-1,1]
                    img_tensor = img_tensor.to(accelerator.device)
                    
                    # 预测
                    logits = model(img_tensor)
                    pred = (torch.sigmoid(logits) > 0.5).float().item()
                    label = sample['is_blur']
                    
                    if pred == label:
                        val_correct += 1
                        if label == 1.0:
                            val_blur_correct += 1
                        else:
                            val_non_blur_correct += 1
                    
                    if label == 1.0:
                        val_blur_total += 1
                    else:
                        val_non_blur_total += 1
            
            val_acc = val_correct / len(val_samples)
            val_blur_acc = val_blur_correct / val_blur_total if val_blur_total > 0 else 0
            val_non_blur_acc = val_non_blur_correct / val_non_blur_total if val_non_blur_total > 0 else 0
            
            accelerator.print(f"\n  Val Accuracy: {val_acc:.4f}")
            accelerator.print(f"  Val Blur Accuracy: {val_blur_acc:.4f} ({val_blur_correct}/{val_blur_total})")
            accelerator.print(f"  Val Non-Blur Accuracy: {val_non_blur_acc:.4f} ({val_non_blur_correct}/{val_non_blur_total})")
            
            model.train()
        
        # 保存最佳模型（基于验证准确率）
        if val_acc > best_acc and accelerator.is_main_process:
            best_acc = val_acc
            save_path = Path(config.output_dir) / "isblur_best.pth"
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(
                unwrapped_model.state_dict(),
                save_path
            )
            accelerator.print(f"✅ 保存最佳模型: {save_path}")
            accelerator.print(f"   准确率: {best_acc:.4f}")
        
        # 每个 epoch 都保存最新模型
        if accelerator.is_main_process:
            save_path = Path(config.output_dir) / f"isblur_epoch_{epoch+1}.pth"
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(
                unwrapped_model.state_dict(),
                save_path
            )
    
    # 训练完成
    accelerator.print("=" * 60)
    accelerator.print("✅ 训练完成！")
    accelerator.print(f"最佳准确率: {best_acc:.4f}")
    accelerator.print(f"模型保存在: {config.output_dir}")
    accelerator.print("=" * 60)


if __name__ == "__main__":
    main()
