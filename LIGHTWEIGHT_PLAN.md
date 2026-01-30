# HYPIR 轻量化训练方案 (A100-40G)

针对 A100-40G 显存限制（原论文使用 A6000-48G），通过对代码库的深入分析，我为您设计了一套完整的轻量化训练方案。主要瓶颈在于判别器骨干网络 (`ConvNeXt-XXLarge`) 和 Stable Diffusion U-Net 的组合开销。

## 1. 判别器轻量化 (最大收益点)

HYPIR 当前使用了极其庞大的 `ConvNeXt-XXLarge` (CLIP) 作为判别器特征提取器。这是显存占用的主要来源之一，对于恢复任务来说可能过度配置了。

### 方案 A: 替换为 ConvNeXt-Large (推荐)
将 `ConvNeXt-XXLarge` 替换为 `ConvNeXt-Large` 或 `ConvNeXt-Base`，可以显著降低显存占用，同时保留 "Vision-Aided" 的语义引导能力。

**修改文件**: `HYPIR/model/backbone.py`

```python:HYPIR/model/backbone.py
 ... existing code ...
class ImageOpenCLIPConvNext(nn.Module):

    def __init__(self, precision="fp32"):
        super().__init__()
        # 修改: 将 "convnext_xxlarge" 替换为 "convnext_large_d_320" 或 "convnext_base"
        # 原版: "convnext_xxlarge", pretrained="laion2b_s34b_b82k_augreg_soup"
        self.model, _, _ = open_clip.create_model_and_transforms(
            "convnext_large_d_320", 
            pretrained="laion2b_s29b_b131k_ft_soup",
            precision=precision,
        )
 ... existing code ...
```

**收益**: 参数量从 ~3B 级别下降到 ~300M 级别，显存节省约 6-10GB。

## 2. 训练策略优化

### 2.1 梯度检查点 (Gradient Checkpointing)
代码中已经实现了梯度检查点逻辑 (`HYPIR/trainer/sd2.py`)，确保在配置文件中开启它。

**修改文件**: `configs/sd2_train.yaml`
```yaml
# 确保此项为 true
gradient_checkpointing: true
```

### 2.2 混合精度训练 (Mixed Precision)
A100 对 `bf16` (BFloat16) 有极佳的支持。代码中已经支持，确保配置正确。

**修改文件**: `configs/sd2_train.yaml`
```yaml
# 推荐使用 bf16 以获得更好的数值稳定性，或者 fp16 以获得更极致的速度/显存
mixed_precision: bf16 
```

### 2.3 调整 Batch Size 与梯度累积
如果上述修改后显存仍不足，可以通过减少 `batch_size` 并增加 `gradient_accumulation_steps` 来模拟大 Batch 训练。

**修改文件**: `configs/sd2_train.yaml`
```yaml
data_config:
  train:
    batch_size: 4  # 从 6 降低到 4 或 2
    
# ...
gradient_accumulation_steps: 2 # 如果 batch_size 减半，则累积步数翻倍
```

## 3. 代码级优化建议

### 3.1 冻结更多参数
`SD2Trainer` 中，`text_encoder` 和 `vae` 已经被冻结。对于判别器，虽然需要训练解码头，但底层的 CLIP backbone 也可以设为 `eval()` 模式（当前代码已设为 `eval` 但可以进一步确保不计算梯度图）。

### 3.2 优化 LoRA Rank
当前的 `lora_rank: 256` 相对较高。对于图像恢复任务，通常 Rank 64 或 128 已经足够。降低 Rank 可以减少可训练参数量和优化器状态（AdamW 需要存储动量，占用 2-3倍 参数量的显存）。

**修改文件**: `configs/sd2_train.yaml`
```yaml
lora_rank: 64 # 从 256 降低
```

## 4. 数据集准备指南

在开始训练之前，您需要准备包含 HQ（高质量）图像路径的 parquet 文件。

### 4.1 制作索引文件
创建一个 Python 脚本（例如 `make_dataset.py`），扫描您的图片文件夹并生成所需的 `.parquet` 文件。

```python:make_dataset.py
import os
import polars as pl  # 需要安装 polars: pip install polars pyarrow

# 1. 设置图片所在目录
image_dir = "/path/to/your/high_quality_images"  # 修改为您的实际路径
image_exts = (".jpg", ".jpeg", ".png", ".webp")

# 2. 递归收集图片路径
image_paths = []
print(f"Scanning images in {image_dir}...")
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.lower().endswith(image_exts):
            # 使用绝对路径
            full_path = os.path.abspath(os.path.join(root, file))
            image_paths.append(full_path)

print(f"Found {len(image_paths)} images.")

# 3. 创建 DataFrame (prompt 设为空字符串，模型会使用空文本条件)
df = pl.from_dict({
    "image_path": image_paths,
    "prompt": [""] * len(image_paths)
})

# 4. 保存为 parquet 文件
output_path = "./dataset_index.parquet"
df.write_parquet(output_path)
print(f"Saved dataset index to {output_path}")
```

### 4.2 配置训练路径
生成 parquet 文件后，修改 `configs/sd2_train.yaml` 指向该文件。

```yaml:configs/sd2_train.yaml
 ...
    dataset:
      target: HYPIR.dataset.realesrgan.RealESRGANDataset
      params:
        file_meta:
          file_list: ./dataset_index.parquet  # 修改为您生成的 parquet 文件路径
          image_path_prefix: ""               # 因为使用了绝对路径，这里留空
          image_path_key: image_path
          prompt_key: prompt
 ...
```

## 总结实施步骤

1.  **优先操作**: 修改 `HYPIR/model/backbone.py`，将 backbone 降级为 `convnext_large_d_320`。
2.  **数据准备**: 运行上述 `make_dataset.py` 脚本生成索引文件。
3.  **配置调整**: 在 `configs/sd2_train.yaml` 中：
    *   将 `dataset.params.file_meta.file_list` 指向您的 parquet 文件。
    *   将 `lora_rank` 降为 64。
    *   将 `batch_size` 调整为 4。
4.  **启动训练**: 使用 `bf16` 模式在 A100 上启动。

这套方案在理论上可以将显存占用控制在 30GB 以内，完全适配 A100-40G。
