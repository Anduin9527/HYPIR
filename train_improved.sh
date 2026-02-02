#!/bin/bash
# HYPIR 改进版训练快速启动脚本

echo "========================================="
echo "  HYPIR Training - Improved Version"
echo "========================================="
echo ""

# 检查conda环境
if [ "$CONDA_DEFAULT_ENV" != "hypir" ]; then
    echo "⚠️  警告: 当前不在 hypir 环境中"
    echo "正在激活 hypir 环境..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate hypir
    if [ $? -ne 0 ]; then
        echo "❌ 错误: 无法激活 hypir 环境"
        exit 1
    fi
fi

echo "✓ Conda 环境: $CONDA_DEFAULT_ENV"
echo ""

# 检查依赖
echo "检查依赖..."
python -c "import pyiqa" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  pyiqa 未安装，正在安装..."
    pip install pyiqa
    if [ $? -ne 0 ]; then
        echo "❌ 错误: 无法安装 pyiqa"
        exit 1
    fi
fi
echo "✓ pyiqa 已安装"

python -c "import swanlab" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  swanlab 未安装，正在安装..."
    pip install swanlab
    if [ $? -ne 0 ]; then
        echo "❌ 错误: 无法安装 swanlab"
        exit 1
    fi
fi
echo "✓ swanlab 已安装"
echo ""

# 检查SwanLab登录状态
echo "检查 SwanLab 登录状态..."
python -c "import swanlab; swanlab.login(relogin=False)" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  SwanLab 未登录"
    echo ""
    read -p "是否现在登录 SwanLab? [y/N]: " login_choice
    if [ "$login_choice" = "y" ] || [ "$login_choice" = "Y" ]; then
        swanlab login
        if [ $? -ne 0 ]; then
            echo "❌ 错误: SwanLab 登录失败"
            echo "提示: 访问 https://swanlab.cn/settings 获取 API Key"
            exit 1
        fi
        echo "✓ SwanLab 登录成功"
    else
        echo "⚠️  警告: 未登录 SwanLab，训练日志将无法上传到云端"
        echo "   如需登录，请运行: swanlab login"
        read -p "是否继续训练? [y/N]: " continue_choice
        if [ "$continue_choice" != "y" ] && [ "$continue_choice" != "Y" ]; then
            exit 0
        fi
    fi
else
    echo "✓ SwanLab 已登录"
fi
echo ""

# 检查验证集
echo "检查验证集..."
VAL_DIR="/data/users/gaoyin/datasets/AIO/Val"
if [ ! -d "$VAL_DIR" ]; then
    echo "❌ 错误: 验证集目录不存在: $VAL_DIR"
    exit 1
fi

declare -a DEG_TYPES=("Blur" "Haze" "Lowlight" "Rain" "Snow")
for deg in "${DEG_TYPES[@]}"; do
    LQ_DIR="$VAL_DIR/$deg/LQ"
    GT_DIR="$VAL_DIR/$deg/GT"
    if [ ! -d "$LQ_DIR" ] || [ ! -d "$GT_DIR" ]; then
        echo "⚠️  警告: $deg 目录结构不完整"
    else
        LQ_COUNT=$(ls -1 "$LQ_DIR"/*.jpg 2>/dev/null | wc -l)
        GT_COUNT=$(ls -1 "$GT_DIR"/*.jpg 2>/dev/null | wc -l)
        echo "  ✓ $deg: LQ=$LQ_COUNT张, GT=$GT_COUNT张"
    fi
done
echo ""

# 询问使用哪个配置
echo "选择训练配置："
echo "  1) 原配置 (sd2_finetune_5k.yaml)"
echo "  2) 改进配置 (sd2_finetune_5k_improved.yaml) [推荐]"
echo ""
read -p "请选择 [1/2]: " choice

case $choice in
    1)
        CONFIG="configs/sd2_finetune_5k.yaml"
        OUTPUT_DIR="./results/baseline"
        echo "使用原配置"
        ;;
    2)
        CONFIG="configs/sd2_finetune_5k_improved.yaml"
        OUTPUT_DIR="./results/improved"
        echo "使用改进配置（推荐）"
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac
echo ""

# 询问输出目录
read -p "输出目录 (默认: $OUTPUT_DIR): " custom_dir
if [ ! -z "$custom_dir" ]; then
    OUTPUT_DIR="$custom_dir"
fi
echo ""

# 确认开始训练
echo "========================================="
echo "训练配置确认："
echo "  配置文件: $CONFIG"
echo "  输出目录: $OUTPUT_DIR"
echo "  验证频率: 每500步"
echo "  最大步数: 50000"
echo "========================================="
echo ""
read -p "确认开始训练? [y/N]: " confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "🚀 开始训练..."
echo ""

# 启动训练
python train.py \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR"

# 检查训练结果
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ 训练完成！"
    echo "========================================="
    echo ""
    echo "结果保存在: $OUTPUT_DIR"
    echo "访问 SwanLab 查看训练曲线: https://swanlab.cn"
    echo ""
else
    echo ""
    echo "========================================="
    echo "❌ 训练失败"
    echo "========================================="
    echo ""
    echo "请检查错误信息并重试"
    exit 1
fi
