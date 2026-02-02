# 推理和评估脚本使用说明

## 文件说明

### 推理脚本
1. **inference.py** - Finetuned 模型推理（主要使用）
2. **inference_baseline.py** - Baseline 模型推理（用于对比）
3. **calculate_lpips.py** - LPIPS 指标计算脚本

### 快捷脚本
1. **quick_inference.sh** - 快速推理（不计算 LPIPS）
2. **quick_inference_with_lpips.sh** - 推理 + LPIPS 评估
3. **run_comparison.sh** - 完整对比实验（Baseline vs Finetuned）

## 使用方法

### 方法 1: 快速推理（不评估）
```bash
bash quick_inference.sh
```

### 方法 2: 推理 + LPIPS 评估
```bash
bash quick_inference_with_lpips.sh
```
这会自动计算 LPIPS 并生成：
- `inference_results/lpips_results.csv` - 详细的每张图 LPIPS 值
- `inference_results/lpips_report.txt` - 统计报告（最大值、最小值、平均值）

### 方法 3: 完整对比实验
```bash
bash run_comparison.sh
```
对比 Baseline 和 Finetuned 两个模型，生成对比报告。

### 方法 4: 手动命令（最灵活）
```bash
# 推理
CUDA_VISIBLE_DEVICES=3 python inference.py \
  --lq_dir /path/to/lq \
  --output_dir ./results \
  --checkpoint ./results/checkpoint-50000 \
  --prompt "high quality, sharp details" \
  --gt_dir /path/to/gt \
  --use_ema

# 单独计算 LPIPS（如果需要）
python calculate_lpips.py \
  --baseline_dir ./results1/result \
  --finetuned_dir ./results2/result \
  --gt_dir /path/to/gt \
  --output_csv ./lpips_comparison.csv
```

## 输出说明

### 图像保存格式
- **格式**: JPEG
- **质量**: 96
- **优化**: 开启
- **色彩空间**: RGB
- **子采样**: 0（最高质量）

### LPIPS 输出
1. **CSV 文件**: 包含每张图的详细指标
   ```
   image_name,baseline_lpips,finetuned_lpips,improvement
   0001,0.123456,0.098765,0.024691
   ...
   ```

2. **统计报告**: 文本格式的汇总统计
   - 最小值、最大值、平均值
   - 改进比例和改进图像数
   - 结论

## 参数说明

### inference.py 主要参数
- `--lq_dir`: 输入低质量图像目录（必需）
- `--output_dir`: 输出目录（必需）
- `--checkpoint`: checkpoint 路径（默认: ./results/checkpoint-50000）
- `--prompt`: 修复 prompt（默认: "high quality, sharp details"）
- `--gt_dir`: GT 目录，用于自动计算 LPIPS（可选）
- `--use_ema`: 使用 EMA 权重（推荐）
- `--skip_lpips`: 跳过 LPIPS 计算
- `--patch_size`: patch 大小（默认: 512）
- `--stride`: 滑动步长（默认: 256）

### calculate_lpips.py 参数
- `--baseline_dir`: Baseline 结果目录
- `--finetuned_dir`: Finetuned 结果目录
- `--gt_dir`: Ground Truth 目录
- `--output_csv`: 输出 CSV 文件路径
- `--device`: 计算设备（默认: cuda）

## 注意事项

1. **LPIPS 越小越好**：LPIPS 衡量感知距离，值越小表示与 GT 越接近
2. **GPU 显存**：推理比训练占用显存少，512x512 约需 5-8GB
3. **处理时间**：每张图约 3-6 秒（取决于图像大小）
4. **文件格式**：输出为高质量 JPEG（quality=96），兼顾质量和文件大小

## 依赖安装

如果 LPIPS 计算失败，安装 pyiqa:
```bash
conda activate hypir
pip install pyiqa
```
