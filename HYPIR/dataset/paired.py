from typing import Dict, Optional, List
import io
import os
import random
import time
import numpy as np
import torch
from torch.utils import data
from PIL import Image
import polars as pl
import cv2

from HYPIR.dataset.utils import augment


class PairedParquetDataset(data.Dataset):
    """
    Dataset for paired image restoration (LQ, HQ) from a parquet file.
    """

    def __init__(
        self,
        file_meta,
        out_size,
        crop_type,
        use_hflip,
        use_rot,
        image_backend_cfg=None,  # Placeholder to match config structure if needed
        return_file_name=False,
    ):
        super(PairedParquetDataset, self).__init__()
        self.file_meta = file_meta
        self.out_size = out_size
        self.crop_type = crop_type
        self.use_hflip = use_hflip
        self.use_rot = use_rot
        self.return_file_name = return_file_name

        # Load file list from parquet
        self.data_rows = self._load_parquet(file_meta)

    def _load_parquet(self, file_meta):
        file_list_path = file_meta["file_list"]
        lq_key = file_meta.get("lq_key", "lq_path")
        gt_key = file_meta.get("gt_key", "gt_path")
        prompt_key = file_meta.get("prompt_key", "prompt")

        df = pl.read_parquet(file_list_path)
        rows = []
        for row in df.iter_rows(named=True):
            rows.append(
                {
                    "lq_path": row[lq_key],
                    "gt_path": row[gt_key],
                    "prompt": row.get(prompt_key, ""),
                }
            )
        return rows

    def _load_image(self, path):
        # Retry logic could be added here if reading from network/s3
        try:
            img = Image.open(path).convert("RGB")
            return np.array(img)
        except Exception as e:
            print(f"Failed to load image {path}: {e}")
            return None

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.data_rows[index]
        lq_path = row["lq_path"]
        gt_path = row["gt_path"]
        prompt = row["prompt"]

        img_lq = self._load_image(lq_path)
        img_gt = self._load_image(gt_path)

        if img_lq is None or img_gt is None:
            # Fallback to random image if load fails
            return self.__getitem__(random.randint(0, len(self) - 1))

        # Augmentation: Random Flip and Rotation
        # We need to apply the SAME transformation to both LQ and GT
        # augment function in utils.py handles lists of images with same random state
        if self.use_hflip or self.use_rot:
            # augment function returns a list if input is a list
            # It internally handles the random state so passing them together works
            # But wait, check utils.augment implementation:
            # It generates random boolean once then applies to all images in list.
            # Perfect.
            [img_lq, img_gt] = augment([img_lq, img_gt], self.use_hflip, self.use_rot)

        if self.out_size is not None and self.crop_type == "none":
            # If not cropping, enforce resize to out_size if dimensions mismatch
            h, w, c = img_lq.shape
            if h != self.out_size or w != self.out_size:
                # Resize both to out_size
                img_lq = cv2.resize(
                    img_lq,
                    (self.out_size, self.out_size),
                    interpolation=cv2.INTER_CUBIC,
                )
                img_gt = cv2.resize(
                    img_gt,
                    (self.out_size, self.out_size),
                    interpolation=cv2.INTER_CUBIC,
                )

        # Preprocessing: [0, 255] -> [0, 1], HWC -> CHW
        img_lq = img_lq.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_gt = img_gt.transpose(2, 0, 1).astype(np.float32) / 255.0

        img_lq = torch.from_numpy(img_lq)
        img_gt = torch.from_numpy(img_gt)

        data = {
            "LQ": img_lq,
            "GT": img_gt,
            "txt": prompt,
        }

        if self.return_file_name:
            data["filename"] = os.path.basename(gt_path)

        return data

    def __len__(self) -> int:
        return len(self.data_rows)
