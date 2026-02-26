# -*- coding: utf-8 -*-
"""
专门为 NAFNet Expert 预训练编写的数据集加载器
继承自 PairedParquetDataset，并在初始化时根据 expert_type 过滤数据
"""

from HYPIR.dataset.paired import PairedParquetDataset


class ExpertDataset(PairedParquetDataset):
    """
    专家预训练数据集
    支持通过给定的 expert_type 来过滤 parquet 数据
    例如：expert_type = 'Rain', 'Blur', 'Lowlight', 'Snow', 'Haze'
    """

    def __init__(
        self,
        file_meta,
        out_size,
        expert_type,
        crop_type="none",
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

        # 保存原始长度供 Logging
        self.original_size = len(self.data_rows)
        self.expert_type = expert_type

        # 过滤只保留对应 expert_type 的样本
        # 匹配方式：由于路径名如 /data/users/.../Rain/... 包含退化名，或者直接有 degradation_type
        filtered_rows = []
        for row in self.data_rows:
            # 兼容：如果配置中读取了 degradation_type 字段则直接匹配，否则通过 lq_path 子串匹配
            if self.expert_type in row.get("degradation_type", row["lq_path"]):
                filtered_rows.append(row)

        self.data_rows = filtered_rows


if __name__ == "__main__":
    config = {
        "file_list": "custom_5k_imp_with_labels.parquet",
    }
    dataset = ExpertDataset(
        file_meta=config,
        out_size=256,
        expert_type="Rain",
    )
    print(f"Total Original Data: (assuming parquet loaded)")
    print(f"Filtered Dataset size for Rain: {len(dataset)}")
