# -*- coding: utf-8 -*-
"""
带有模糊标注的数据集，用于 isBlur 分类器训练
继承自 PairedParquetDataset，添加 is_blur 标签
"""
from HYPIR.dataset.paired import PairedParquetDataset
import torch


class BlurLabeledDataset(PairedParquetDataset):
    """
    带有模糊标注的配对数据集
    
    在原有的 LQ/GT 对基础上，添加 is_blur 标签（0=非模糊，1=模糊）
    标签从路径中提取（检查是否包含 'Blur' 关键词）
    """
    
    def __init__(
        self,
        file_meta,
        out_size,
        crop_type='none',
        use_hflip=True,
        use_rot=False,
        image_backend_cfg=None,
        return_file_name=False,
    ):
        super().__init__(
            file_meta=file_meta,
            out_size=out_size,
            crop_type=crop_type,
            use_hflip=use_hflip,
            use_rot=use_rot,
            image_backend_cfg=image_backend_cfg,
            return_file_name=return_file_name,
        )
    
    def __getitem__(self, index: int):
        """
        返回数据项，包含 is_blur 标签
        
        Returns:
            dict: {
                'LQ': [3, H, W] tensor,
                'GT': [3, H, W] tensor,
                'txt': str (prompt),
                'is_blur': scalar tensor (0 或 1),
                'filename': str (可选)
            }
        """
        # 调用父类方法获取基础数据
        data = super().__getitem__(index)
        
        # 从路径中提取 is_blur 标签
        lq_path = self.data_rows[index]['lq_path']
        is_blur = 1.0 if 'Blur' in lq_path else 0.0
        data['is_blur'] = torch.tensor(is_blur, dtype=torch.float32)
        
        return data


if __name__ == "__main__":
    # 简单测试
    from omegaconf import OmegaConf
    
    # 创建测试配置
    config = {
        'file_list': 'custom_5k_with_labels.parquet',
    }
    
    dataset = BlurLabeledDataset(
        file_meta=config,
        out_size=512,
        crop_type='none',
        use_hflip=False,
        use_rot=False,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # 测试读取
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"LQ shape: {sample['LQ'].shape}")
        print(f"GT shape: {sample['GT'].shape}")
        print(f"is_blur: {sample['is_blur'].item()}")
        print(f"txt: {sample['txt']}")
