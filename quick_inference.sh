#!/bin/bash
# 快速推理测试脚本

# 设置 GPU
export CUDA_VISIBLE_DEVICES=3

# 配置参数
LQ_DIR="/data/users/gaoyin/datasets/AIO/Validation/LQ"  # 使用绝对路径
OUTPUT_DIR="./results/inference_finetune_imp"  # 输出文件夹
CHECKPOINT="./results/checkpoint-50000"  # checkpoint 路径
PROMPT="high quality, sharp details"   # 可选的 prompt

# 运行推理
python inference.py \
  --lq_dir "$LQ_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --checkpoint "$CHECKPOINT" \
  --prompt "$PROMPT" \
  --use_ema \

