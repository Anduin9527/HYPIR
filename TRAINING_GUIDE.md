# HYPIR 训练指南

完整的训练配置、验证和优化指南。

## 快速开始

### 1. 环境准备

```bash
conda activate hypir
pip install -r requirements.txt
```

### 2. SwanLab 设置（首次使用必需）

```bash
swanlab login
```

输入你的API Key（从 https://swanlab.cn/settings 获取）。

### 3. 测试SwanLab连接（推荐）

```bash
python test_swanlab.py
```

### 4. 开始训练

**使用快速启动脚本（推荐）：**
```bash
./train_improved.sh
```

**或手动启动：**
```bash
# 使用改进配置（推荐）
python train.py --config configs/sd2_finetune_5k_improved.yaml

# 使用原配置
python train.py --config configs/sd2_finetune_5k.yaml
```

---

## 目录

- [最新改进](#最新改进)
- [配置说明](#配置说明)
- [SwanLab监控](#swanlab监控)
- [验证设置](#验证设置)
- [训练策略优化](#训练策略优化)
- [故障排除](#故障排除)

---

## 最新改进

### ✅ 1. 使用 pyiqa 库计算评估指标

- 替换了 `scikit-image`，使用 `pyiqa` 计算 PSNR、SSIM 和 LPIPS
- 支持 Y 通道计算（符合标准图像质量评估）
- GPU 加速，计算更快

### ✅ 2. 验证全部图片

- 每次验证评估全部验证集图片（每类约100张，共约500张）
- 配置: `num_samples_per_type: ~` (None)
- 更准确的评估结果

### ✅ 3. SwanLab 完整集成

- 自动初始化和日志记录
- 实验名称使用配置文件名（如 `sd2_finetune_5k_improved`）
- 实时上传训练和验证指标

### ✅ 4. 添加 SSIM 损失

- 在训练中直接优化 SSIM 指标
- 配置: `lambda_ssim: 1.0`

### ✅ 5. 损失权重优化

针对 PSNR/SSIM 不高的问题，调整了损失权重：
- `lambda_gan: 0.3` (↓ 从 0.5)
- `lambda_lpips: 3.0` (↓ 从 5.0)
- `lambda_l2: 2.0` (↑ 从 1.0)
- `lambda_l1: 2.0` (↑ 从 1.0)

---

## 配置说明

### 配置文件对比

| 配置项 | 原配置 (sd2_finetune_5k) | 改进配置 (sd2_finetune_5k_improved) |
|--------|--------------------------|-----------------------------------|
| lambda_gan | 0.5 | 0.3 ⬇️ |
| lambda_lpips | 5.0 | 3.0 ⬇️ |
| lambda_l2 | 1.0 | 2.0 ⬆️ |
| lambda_l1 | 1.0 | 2.0 ⬆️ |
| lambda_ssim | - | 1.0 ✨ |
| lr_G | 1e-5 | 2e-5 ⬆️ |
| lr_D | 1e-5 | 5e-6 ⬇️ |
| weight_decay | - | 1e-4 ✨ |

### 验证配置

```yaml
data_config:
  val:
    val_dir: /data/users/gaoyin/datasets/AIO/Val
    degradation_types: [Blur, Haze, Lowlight, Rain, Snow]
    batch_size: 1
    num_samples_per_type: ~  # None = 全部验证

validation_steps: 500  # 每500步验证一次
```

### 验证集目录结构

```
/data/users/gaoyin/datasets/AIO/Val/
├── Blur/
│   ├── LQ/*.jpg
│   └── GT/*.jpg
├── Haze/
│   ├── LQ/*.jpg
│   └── GT/*.jpg
├── Lowlight/
│   ├── LQ/*.jpg
│   └── GT/*.jpg
├── Rain/
│   ├── LQ/*.jpg
│   └── GT/*.jpg
└── Snow/
    ├── LQ/*.jpg
    └── GT/*.jpg
```

**注意：** LQ 和 GT 目录中的文件名必须一一对应。

---

## SwanLab监控

### 初始化

训练开始时会自动初始化SwanLab，终端显示：
```
Initialized SwanLab tracker: project=HYPIR-training, experiment=sd2_finetune_5k_improved
```

实验名称直接使用配置文件名，方便识别。

### 记录的指标

#### 训练损失（每步）
- `loss/G_total` - 生成器总损失
- `loss/G_mse` - MSE损失
- `loss/G_l1` - L1损失
- `loss/G_lpips` - LPIPS损失
- `loss/G_ssim` - SSIM损失（如果启用）
- `loss/G_disc` - GAN损失
- `loss/D` - 判别器损失
- `loss/D_logits_real` - 真实样本logits
- `loss/D_logits_fake` - 生成样本logits

#### 验证指标（每validation_steps步）
- `val/PSNR` - 平均PSNR
- `val/SSIM` - 平均SSIM
- `val/LPIPS` - 平均LPIPS
- `val/Final_Score` - 综合评分 (PSNR + 10*SSIM - 5*LPIPS)

#### 按退化类型的指标
- `val/{Blur,Haze,Lowlight,Rain,Snow}/{PSNR,SSIM,LPIPS}`

#### 梯度范数（每log_grad_steps步）
- `grad_norm/conv_out_{l2,l1,lpips,disc}`

### 查看监控

访问 https://swanlab.cn 实时查看：
- 训练损失曲线
- 验证指标变化
- 不同退化类型的表现
- 多个实验的对比

---

## 训练策略优化

### 问题分析

如果你的训练结果是：
- ❌ PSNR 和 SSIM 不高
- ✅ LPIPS 较好

**原因：**
1. GAN损失权重过高 → 过度追求视觉真实感，牺牲像素精度
2. LPIPS损失权重过高 → 感知损失与像素损失冲突
3. 缺乏SSIM直接优化 → 结构相似性不足
4. 学习率不平衡 → 判别器过强

### 改进方案

改进配置 (`sd2_finetune_5k_improved.yaml`) 通过以下策略优化：

#### 1. 降低感知损失主导
```yaml
lambda_gan: 0.3        # 降低GAN权重
lambda_lpips: 3.0      # 降低LPIPS权重
```

#### 2. 增强像素级监督
```yaml
lambda_l2: 2.0         # 提高MSE权重，直接优化PSNR
lambda_l1: 2.0         # 提高L1权重，增强边缘细节
```

#### 3. 直接优化目标指标
```yaml
lambda_ssim: 1.0       # 新增SSIM损失
```

#### 4. 平衡生成器和判别器
```yaml
lr_G: 2e-5             # 提高生成器学习率
lr_D: 5e-6             # 降低判别器学习率
```

#### 5. 防止过拟合
```yaml
weight_decay: 1e-4     # 添加权重衰减
```

### 预期效果

使用改进配置后：
- **PSNR**: +1~2 dB 提升
- **SSIM**: +0.02~0.05 提升
- **LPIPS**: 略微上升但仍保持较好水平
- **Final_Score**: 显著提升

### 进阶优化

如果效果仍不理想，可以尝试：

#### 分阶段训练

**阶段1（0-10K steps）：**
```yaml
lambda_l2: 3.0
lambda_l1: 3.0
lambda_lpips: 2.0
lambda_gan: 0.2
```
目标：快速建立像素级基础

**阶段2（10K-50K steps）：**
使用改进配置，平衡像素和感知损失

#### 调整时间步
```yaml
model_t: 100           # 降低时间步，减少扩散噪声
coeff_t: 100
```

可能提高像素级精度，但可能降低生成多样性。

#### 增加模型容量
```yaml
lora_rank: 512         # 从256增加到512
```

注意：会增加显存占用和训练时间。

---

## 验证设置

### 验证频率

默认每500步验证一次。如果验证时间过长，可以调整：

```yaml
validation_steps: 1000  # 改为每1000步
```

或减少验证样本数：

```yaml
data_config:
  val:
    num_samples_per_type: 50  # 每类只验证50张
```

### 验证时间估计

- 全量验证（500张）：约 5-8 分钟/次
- 减半验证（250张）：约 2-4 分钟/次

### 指标说明

- **PSNR (Y)**: 在Y通道（亮度）上计算峰值信噪比
- **SSIM (Y)**: 在Y通道上计算结构相似性
- **LPIPS**: 使用VGG感知损失网络
- **Final_Score**: `PSNR + 10 * SSIM - 5 * LPIPS`

---

## 故障排除

### SwanLab 相关

#### 问题1: SwanLab未初始化

**症状：** 训练正常，但SwanLab网页无数据

**解决：**
1. 确认已运行 `swanlab login`
2. 检查配置: `report_to: swanlab`
3. 查看日志是否有 "Initialized SwanLab tracker"

#### 问题2: API Key错误

```bash
swanlab login
```

#### 问题3: 指标未上传

检查网络连接，等待几分钟（可能有延迟）。

### 训练相关

#### 问题1: 验证时显存不足

减少验证样本：
```yaml
num_samples_per_type: 30
```

#### 问题2: SSIM损失报错

确认：
- pyiqa 已安装: `pip install pyiqa`
- `init_metrics()` 被调用
- 配置中有 `lambda_ssim`

#### 问题3: 训练速度慢

- 减少验证频率: `validation_steps: 1000`
- 减少日志频率: `log_image_steps: 200`
- 使用更少的验证样本

### 验证集相关

#### 问题1: 验证集目录不存在

确认路径：
```bash
ls -la /data/users/gaoyin/datasets/AIO/Val
```

#### 问题2: LQ/GT文件不匹配

确保文件名一一对应：
```bash
ls /data/users/gaoyin/datasets/AIO/Val/Blur/LQ/
ls /data/users/gaoyin/datasets/AIO/Val/Blur/GT/
```

---

## 训练时间估计

基于 A100 40GB，50K steps：
- **原配置**: 约 8-10 小时
- **改进配置**: 约 9-12 小时（+SSIM损失）
- **验证时间**（500张）: 约 5-8 分钟/次

---

## 对比实验建议

同时运行两个实验进行对比：

```bash
# 终端1: 基线实验
python train.py --config configs/sd2_finetune_5k.yaml \
  --output_dir ./results/baseline

# 终端2: 改进实验
python train.py --config configs/sd2_finetune_5k_improved.yaml \
  --output_dir ./results/improved
```

在SwanLab中对比两个实验的指标曲线。

---

## 文件说明

- `train.py` - 训练脚本
- `train_improved.sh` - 快速启动脚本
- `test_swanlab.py` - SwanLab连接测试
- `configs/sd2_finetune_5k.yaml` - 原始配置
- `configs/sd2_finetune_5k_improved.yaml` - 改进配置
- `INFERENCE_README.md` - 推理使用说明

---

## 总结

**核心改进策略：**
1. ✅ 降低GAN和LPIPS权重
2. ✅ 提高L1/L2权重
3. ✅ 添加SSIM损失
4. ✅ 调整学习率比例
5. ✅ 使用pyiqa计算指标
6. ✅ SwanLab完整集成

**建议优先级：**
1. 使用改进配置训练完整模型
2. 在验证集上评估效果
3. 根据结果微调超参数
4. 考虑分阶段训练策略

祝训练顺利！🚀

有问题随时查看 SwanLab 监控或检查训练日志。
