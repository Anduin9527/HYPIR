import unittest
import os
import shutil
import numpy as np
import torch
from PIL import Image
import polars as pl
from HYPIR.dataset.paired import PairedParquetDataset


class TestPairedDataset(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_data"
        os.makedirs(self.test_dir, exist_ok=True)

        # Create dummy images
        self.lq_path = os.path.join(self.test_dir, "lq.png")
        self.gt_path = os.path.join(self.test_dir, "gt.png")

        # Create different patterns to verify flip/rot
        lq_img = np.zeros((100, 100, 3), dtype=np.uint8)
        lq_img[0:50, :, :] = 255  # Top half white
        Image.fromarray(lq_img).save(self.lq_path)

        gt_img = np.zeros((100, 100, 3), dtype=np.uint8)
        gt_img[0:50, :, :] = 255  # Top half white
        Image.fromarray(gt_img).save(self.gt_path)

        # Create dummy parquet
        self.parquet_path = os.path.join(self.test_dir, "meta.parquet")
        df = pl.DataFrame(
            {
                "lq_path": [self.lq_path],
                "gt_path": [self.gt_path],
                "prompt": ["test prompt"],
            }
        )
        df.write_parquet(self.parquet_path)

        self.file_meta = {
            "file_list": self.parquet_path,
            "lq_key": "lq_path",
            "gt_key": "gt_path",
            "prompt_key": "prompt",
        }

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_load_and_augment(self):
        dataset = PairedParquetDataset(
            file_meta=self.file_meta,
            out_size=100,
            crop_type="none",
            use_hflip=True,  # Force True for testing? augment uses random...
            use_rot=True,
            return_file_name=False,
        )

        # Disable randomness for testing or test multiple times?
        # Since augment is random, let's just check shapes and value ranges for now
        # and ensure LQ and GT are consistent in terms of transformation if we can.
        # But hard to check consistency without mocking random.

        item = dataset[0]
        self.assertIn("LQ", item)
        self.assertIn("GT", item)
        self.assertIn("txt", item)

        lq = item["LQ"]
        gt = item["GT"]

        self.assertEqual(lq.shape, (3, 100, 100))
        self.assertEqual(gt.shape, (3, 100, 100))
        self.assertEqual(item["txt"], "test prompt")

        # Check normalization
        self.assertTrue(lq.max() <= 1.0)
        self.assertTrue(lq.min() >= 0.0)

        # Check synchronization - if we flip, both should flip.
        # Since our image has top half white, if we rotate 90 deg, usually left or right becomes white.
        # Original: Top half (y < 50) is 1.0.
        # If no flip/rot: lq[:, 0:50, :] should be close to 1.

        # Let's run multiple times and see if they always match
        for _ in range(10):
            item = dataset[0]
            lq = item["LQ"]
            gt = item["GT"]

            # They should be identical since input images are identical
            # and augmentations are synchronized
            diff = torch.abs(lq - gt).sum()
            self.assertLess(diff, 1e-5, "LQ and GT should be transformed identically")


if __name__ == "__main__":
    unittest.main()
