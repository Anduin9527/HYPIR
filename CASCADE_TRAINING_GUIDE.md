# 级联去模糊训练指南

## 概述

本指南介绍如何使用两阶段级联去模糊方案来改善 HYPIR 对模糊图像的处理效果。

### 核心思路

从训练曲线来看，HYPIR 对 Blur 类型图像的 PSNR 表现不佳（初期波动大，最终稳定在 18.3 左右），而其他退化类型（Haze/Lowlight/Rain/Snow）表现良好。因此采用**解耦训练**策略：

1. **阶段 1**：单独训练去模糊模块（isBlur + NAFNet）
2. **阶段 2**：将去模糊模块作为冻结的预处理，让 HYPIR 专注于非模糊退化

### 架构流程

```
阶段 1：去模糊模块训练
  └─ 1a: 训练 isBlur 分类器 (准确率 > 95%)
  └─ 1b: 冻结 isBlur，微调 NAFNet (PSNR 提升 > 3dB)
  └─ 输出: deblur_module_best.pth

阶段 2：HYPIR 级联训练
  ├─ 加载冻结的去模糊模块
  ├─ Blur 数据: LQ → NAFNet → HYPIR
  └─ 其他数据: LQ → HYPIR
```

---

## 准备工作

### 1. 环境配置

确保 conda 环境已激活，并安装了所有依赖：

```bash
conda activate hypir
pip install -r requirements.txt
```

### 2. 准备 NAFNet 预训练权重

级联方案需要 NAFNet 的预训练权重。请将权重放置在：

```bash
/data/users/gaoyin/pretrained/nafnet_gopro.pth
```

**获取方式**：

- 官方 GitHub: https://github.com/megvii-research/NAFNet
- 推荐使用 GoPro 数据集上的预训练权重

### 3. 数据标注

为训练数据添加退化类型标注（从路径中自动提取）：

```bash
python scripts/add_degradation_label.py --input custom_5k.parquet --output custom_5k_with_labels.parquet
```

**输出示例**：

```
退化类型统计:
  Blur        : 1000 样本
  Haze        : 1000 样本
  Lowlight    : 1000 样本
  Rain        : 1000 样本
  Snow        : 1000 样本
```

### 4. SwanLab 设置（首次使用必需）

本项目使用 SwanLab 进行训练监控和实验跟踪。首次使用需要进行设置：

**登录 SwanLab**：

```bash
swanlab login
```

系统会提示输入 API Key。前往 https://swanlab.cn/settings 获取你的 API Key。

**测试连接（推荐）**：

```bash
python -c "import swanlab; swanlab.init(project='test', experiment_name='connection_test'); swanlab.finish()"
```

如果没有报错，说明 SwanLab 配置成功。

**实验监控**：

训练过程中，SwanLab 会自动记录：
- 训练损失: `train/loss`, `train/loss_l1`, `train/loss_l2`, `train/loss_gan`, `train/loss_lpips`
- 验证指标: `val/blur_psnr`, `val/haze_psnr`, `val/psnr`, `val/ssim`, `val/lpips`, `val/final_score`
- 实验名称使用配置文件名（如 `isblur_pretrain`, `nafnet_finetune`, `sd2_cascade_blur`）

查看实验结果：登录 https://swanlab.cn/，在项目列表中找到对应的实验。

---

## 阶段 1：去模糊模块训练

### Step 1a：训练 isBlur 分类器

**目标**：训练一个准确率 > 95% 的模糊检测器

**命令**：

```bash
python train_isblur.py --config configs/isblur_pretrain.yaml
```

**配置说明** (`configs/isblur_pretrain.yaml`)：

```yaml
output_dir: ./results/isblur_pretrain

data:
  file_list: custom_5k_with_labels.parquet
  image_size: 512
  batch_size: 32  # ResNet18 占用内存较小
  num_workers: 4

backbone: resnet18  # 'resnet18' 或 'efficientnet_b0'
learning_rate: 1e-3
weight_decay: 1e-4
num_epochs: 10
mixed_precision: bf16
```

**预期结果**：

- **训练时间**: ~30 分钟（10 epochs）
- **准确率**: > 95%
- **输出**: `./results/isblur_pretrain/isblur_best.pth`

**监控指标**：

```
Epoch 10: Loss=0.0234, Accuracy=0.9712
  - Blur Accuracy: 0.9801 (980/1000)
  - Non-Blur Accuracy: 0.9675 (3870/4000)
```

---

### Step 1b：微调 NAFNet

**目标**：在 Blur 数据上微调 NAFNet，PSNR 提升 > 3dB

**命令**：

```bash
python train_deblur.py --config configs/nafnet_finetune_blur.yaml
```

**配置说明** (`configs/nafnet_finetune_blur.yaml`)：

```yaml
output_dir: ./results/nafnet_finetune

data:
  file_list: custom_5k_with_labels.parquet
  image_size: 512
  batch_size: 8  # NAFNet 内存占用较大
  num_workers: 4

isblur:
  backbone: resnet18
  checkpoint: ./results/isblur_pretrain/isblur_best.pth  # 阶段 1a 输出

nafnet:
  checkpoint: /data/users/gaoyin/pretrained/nafnet_gopro.pth  # 预训练权重
  width: 64
  enc_blks: [2, 2, 4, 8]
  middle_blk_num: 12
  dec_blks: [2, 2, 2, 2]

learning_rate: 1e-4  # 微调用较小学习率
lambda_l1: 1.0
lambda_l2: 1.0
num_epochs: 20
gradient_accumulation_steps: 2
```

**预期结果**：

- **训练时间**: ~2-3 小时（20 epochs，~1000 Blur 样本）
- **PSNR 提升**: 18.3 → 21+ dB
- **输出**: `./results/nafnet_finetune/deblur_module_best.pth`（包含 isBlur + NAFNet）

**监控指标**：

```
Epoch 20: Loss=0.0123 (L1: 0.0089, L2: 0.0034)
  Val PSNR: 21.45 dB
  Val SSIM: 0.7823
```

---

## 阶段 2：HYPIR 级联训练

### 目标

在 HYPIR 训练中集成去模糊预处理：

- **Blur 数据**：LQ → NAFNet → HYPIR
- **其他数据**：LQ → HYPIR
- 去模糊模块**冻结**，不参与梯度更新

### 命令

```bash
python train.py --config configs/sd2_cascade_blur.yaml
```

### 配置说明 (`configs/sd2_cascade_blur.yaml`)

**关键配置**：

```yaml
base_model_type: cascade_sd2  # 使用级联训练器

# 去模糊模块（阶段 1 输出）
deblur_module_checkpoint: ./results/nafnet_finetune/deblur_module_best.pth
blur_threshold: 0.5

# 数据集（需要包含 degradation_type）
data_config:
  train:
    dataset:
      params:
        file_meta:
          file_list: custom_5k_with_labels.parquet
```

**完整配置请参考 `configs/sd2_cascade_blur.yaml`**

### 预期效果

| 指标              | 原始 HYPIR | 级联方案      | 提升         |
| --------------- | -------- | --------- | ---------- |
| Blur PSNR       | 18.3     | 20-22     | +2-4 dB    |
| Blur SSIM       | 0.70     | 0.75-0.80 | +0.05-0.10 |
| Blur LPIPS      | 0.15     | 0.12-0.14 | -0.01-0.03 |
| Haze/Rain/Snow  | 保持       | 保持或略升     | -          |
| Overall Final_Score | 基线     | 提升 5-10%  | -          |

### 监控

使用 SwanLab 查看训练指标：

```bash
# 项目: HYPIR-cascade
# 关键指标:
#   - val/Blur/PSNR
#   - val/Blur/SSIM
#   - val/Final_Score
```

---

## 推理使用

### 不使用去模糊（原始 HYPIR）

```bash
python inference.py \
    --lq_dir ./test_images \
    --output_dir ./results_hypir \
    --checkpoint ./results/sd2_finetune_5k/checkpoint-50000
```

### 使用去模糊预处理（级联）

```bash
python inference.py \
    --lq_dir ./test_images \
    --output_dir ./results_cascade \
    --checkpoint ./results/cascade_training/checkpoint-50000 \
    --use_deblur \
    --deblur_checkpoint ./results/nafnet_finetune/deblur_module_best.pth \
    --blur_threshold 0.5
```

**参数说明**：

- `--use_deblur`: 启用去模糊预处理
- `--deblur_checkpoint`: 去模糊模块路径（阶段 1 输出）
- `--blur_threshold`: isBlur 分类阈值（默认 0.5）

**推理流程**：

1. 对每张输入图像，运行 isBlur 分类器
2. 如果 `isBlur_prob > threshold`，应用 NAFNet 去模糊
3. 将（去模糊后的）图像送入 HYPIR 增强

### 快捷脚本

为方便使用，项目提供了快捷推理脚本：

**快速推理（不计算指标）**：

```bash
bash quick_inference.sh
```

**推理 + 指标评估**：

如果有 Ground Truth 图像，可以计算 PSNR、SSIM、LPIPS 指标：

```bash
python inference.py \
    --lq_dir /path/to/lq \
    --output_dir ./results \
    --checkpoint ./results/cascade_training/checkpoint-50000 \
    --gt_dir /path/to/gt \
    --use_deblur \
    --deblur_checkpoint ./results/nafnet_finetune/deblur_module_best.pth
```

输出会包含：
- 恢复后的图像（保存在 `--output_dir`）
- 评估报告（包含各退化类型的 PSNR/SSIM/LPIPS/Final_Score）

### 输出说明

**图像格式**：
- 格式: JPEG
- 质量: 96
- 优化: 开启
- 命名: 与输入文件名相同

**评估指标**：

如果提供了 `--gt_dir`，会自动计算：
- **PSNR (Y)**: 峰值信噪比（Y 通道）
- **SSIM (Y)**: 结构相似性（Y 通道）
- **LPIPS**: 感知相似性（值越小越好）
- **Final_Score**: PSNR(Y) + 10×SSIM(Y) - 5×LPIPS

---

## 完整训练流程（快速参考）

```bash
# 激活环境
conda activate hypir

# Step 0: 数据标注
python scripts/add_degradation_label.py \
    --input custom_5k.parquet \
    --output custom_5k_with_labels.parquet

# Step 1a: 训练 isBlur 分类器（~30 分钟）
python train_isblur.py --config configs/isblur_pretrain.yaml

# Step 1b: 微调 NAFNet（~2-3 小时）
python train_deblur.py --config configs/nafnet_finetune_blur.yaml

# Step 2: HYPIR 级联训练（~10-20 小时）
python train.py --config configs/sd2_cascade_blur.yaml

# 推理测试
python inference.py \
    --lq_dir ./test_images \
    --output_dir ./results_cascade \
    --checkpoint ./results/cascade_training/checkpoint-50000 \
    --use_deblur \
    --deblur_checkpoint ./results/nafnet_finetune/deblur_module_best.pth
```

---

## 故障排除

### 问题 1：NAFNet 导入失败（已解决）

**说明**：

NAFNet 已完全迁移到 HYPIR 项目本地（`HYPIR/model/nafnet.py`），不再依赖外部 EVSSM2 项目。如果遇到导入问题，请确保文件存在：

```bash
ls HYPIR/model/nafnet.py
ls HYPIR/model/nafnet_wrapper.py
```

这些文件包含了 NAFNet 的完整实现，无需额外依赖。

---

### 问题 2：去模糊模块加载失败

**错误**：
```
KeyError: 'isblur' or 'nafnet'
```

**解决方案**：

检查 checkpoint 文件完整性：

```python
import torch
ckpt = torch.load('./results/nafnet_finetune/deblur_module_best.pth')
print(ckpt.keys())  # 应包含: ['isblur', 'nafnet', 'config']
```

如果缺少键，需要重新运行阶段 1b。

---

### 问题 3：内存不足

**错误**：
```
RuntimeError: CUDA out of memory
```

**解决方案**：

调整配置：

```yaml
# configs/nafnet_finetune_blur.yaml
data:
  batch_size: 4  # 减小 batch size (原 8)
gradient_accumulation_steps: 4  # 增加梯度累积 (原 2)
```

或者：

```yaml
# configs/sd2_cascade_blur.yaml
data_config:
  train:
    batch_size: 2  # 减小 batch size (原 4)
gradient_accumulation_steps: 2  # 增加梯度累积 (原 1)
```

---

### 问题 4：isBlur 准确率低（< 95%）

**可能原因**：

1. 数据标注不准确（路径中没有 'Blur' 关键词）
2. 训练 epochs 不够
3. 学习率不合适

**解决方案**：

1. 检查数据标注：
   ```python
   import polars as pl
   df = pl.read_parquet('custom_5k_with_labels.parquet')
   print(df.group_by('degradation_type').count())
   ```

2. 增加训练 epochs：
   ```yaml
   num_epochs: 15  # 从 10 增加到 15
   ```

3. 调整学习率：
   ```yaml
   learning_rate: 5e-4  # 降低学习率
   ```

---

### 问题 5：SwanLab 未初始化或指标未上传

**症状**：训练正常运行，但 SwanLab 网页无数据显示

**解决方案**：

1. 确认已登录 SwanLab：
   ```bash
   swanlab login
   ```
   如果未登录或 API Key 错误，重新登录并输入正确的 Key。

2. 检查配置文件是否包含 SwanLab 设置：
   ```yaml
   report_to: swanlab
   swanlab_project: HYPIR-cascade  # 或对应项目名
   ```

3. 检查训练日志是否有 "Initialized SwanLab tracker" 字样。

4. 网络问题：如果指标未上传，检查网络连接，等待几分钟（可能有延迟）。

---

### 问题 6：验证集文件不匹配

**错误**：
```
FileNotFoundError: GT file not found for ...
```

**解决方案**：

确保 LQ 和 GT 文件名一一对应：

```bash
# 检查验证集结构
ls /data/users/gaoyin/datasets/AIO/Val/Blur/LQ/
ls /data/users/gaoyin/datasets/AIO/Val/Blur/GT/

# 文件名应该相同（除了路径中的 LQ/GT 部分）
# 例如: Blur/LQ/001.png 对应 Blur/GT/001.png
```

如果文件名不匹配，需要重新组织数据集或修改数据加载逻辑。

---

## 性能优化建议

### 1. 数据增强

如果 Blur 样本较少，可以增加数据增强：

```yaml
# configs/nafnet_finetune_blur.yaml
# 修改 HYPIR/trainer/deblur_trainer.py 中的数据集创建
use_hflip: true
use_rot: true
```

### 2. 学习率调度

添加学习率衰减：

```python
# 在 DeblurTrainer 中添加
from torch.optim.lr_scheduler import CosineAnnealingLR

self.scheduler = CosineAnnealingLR(
    self.optimizer, 
    T_max=config.num_epochs, 
    eta_min=1e-6
)
```

### 3. 多尺度训练

对 NAFNet 使用多尺度训练：

```yaml
data:
  image_size: [256, 384, 512]  # 随机选择尺寸
```

---

## 参考资料

### 相关文件

- **模型**: `HYPIR/model/isblur.py`, `HYPIR/model/nafnet_wrapper.py`
- **数据集**: `HYPIR/dataset/blur_labeled.py`
- **训练器**: `HYPIR/trainer/deblur_trainer.py`, `HYPIR/trainer/cascade_sd2.py`
- **配置**: `configs/isblur_pretrain.yaml`, `configs/nafnet_finetune_blur.yaml`, `configs/sd2_cascade_blur.yaml`
- **脚本**: `train_isblur.py`, `train_deblur.py`, `train.py`, `inference.py`

### 论文引用

- **NAFNet**: Chen et al. "Simple Baselines for Image Restoration" (ECCV 2022)
- **HYPIR**: 基于 Stable Diffusion 2.1 的图像修复框架

---

## 联系与支持

如有问题，请参考：

- 主 README: `README.md`
- 推理指南: `INFERENCE_README.md`
- 训练指南: `TRAINING_GUIDE.md`

---

**祝训练顺利！🚀**
