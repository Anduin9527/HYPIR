# -*- coding: utf-8 -*-
"""
数据标注脚本：为 parquet 文件添加 degradation_type 列
从 lq_path 中提取退化类型（Blur, Haze, Lowlight, Rain, Snow）
"""
import polars as pl
from pathlib import Path
import argparse


def extract_degradation(path):
    """从路径中提取退化类型"""
    degradation_types = ['Blur', 'Haze', 'Lowlight', 'Rain', 'Snow']
    
    for deg_type in degradation_types:
        if deg_type in path:
            return deg_type
    
    return 'Unknown'


def add_degradation_labels(input_parquet, output_parquet=None):
    """
    为 parquet 文件添加 degradation_type 和 is_blur 列
    
    Args:
        input_parquet: 输入的 parquet 文件路径
        output_parquet: 输出的 parquet 文件路径（默认为 *_with_labels.parquet）
    """
    input_path = Path(input_parquet)
    
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_parquet}")
    
    # 默认输出路径
    if output_parquet is None:
        output_parquet = input_path.parent / f"{input_path.stem}_with_labels.parquet"
    else:
        output_parquet = Path(output_parquet)
    
    print("=" * 60)
    print(f"输入文件: {input_parquet}")
    print(f"输出文件: {output_parquet}")
    print("=" * 60)
    
    # 读取 parquet
    print("读取数据...")
    df = pl.read_parquet(input_parquet)
    print(f"总样本数: {len(df)}")
    
    # 检查必需的列
    if 'lq_path' not in df.columns:
        raise ValueError("数据中缺少 'lq_path' 列")
    
    # 添加 degradation_type 列
    print("提取退化类型...")
    df = df.with_columns([
        pl.col('lq_path').map_elements(
            extract_degradation, 
            return_dtype=pl.Utf8
        ).alias('degradation_type')
    ])
    
    # 添加 is_blur 列（用于 isBlur 分类器训练）
    print("添加 is_blur 标签...")
    df = df.with_columns([
        pl.when(pl.col('degradation_type') == 'Blur')
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias('is_blur')
    ])
    
    # 统计信息
    print("\n退化类型统计:")
    print("-" * 60)
    stats = df.group_by('degradation_type').agg(pl.count().alias('count')).sort('degradation_type')
    for row in stats.iter_rows(named=True):
        print(f"  {row['degradation_type']:<12}: {row['count']:>5} 样本")
    
    # 保存
    print("\n保存标注后的数据...")
    df.write_parquet(output_parquet)
    print(f"✅ 完成！保存至: {output_parquet}")
    print("=" * 60)
    
    # 验证
    print("\n验证结果:")
    df_check = pl.read_parquet(output_parquet)
    print(f"  - 总样本数: {len(df_check)}")
    print(f"  - 列: {df_check.columns}")
    print(f"  - Blur 样本数: {(df_check['is_blur'] == 1).sum()}")
    print(f"  - 非Blur 样本数: {(df_check['is_blur'] == 0).sum()}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="为 parquet 文件添加退化类型标注")
    parser.add_argument(
        "--input", 
        type=str, 
        default="custom_5k.parquet",
        help="输入的 parquet 文件路径"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="输出的 parquet 文件路径（默认: <input>_with_labels.parquet）"
    )
    
    args = parser.parse_args()
    
    try:
        add_degradation_labels(args.input, args.output)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
