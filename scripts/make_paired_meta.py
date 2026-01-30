import os
import argparse
import polars as pl
from glob import glob
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Create paired dataset metadata parquet file."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Path to the dataset root (e.g. AIO/Train).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the output parquet file.",
    )
    parser.add_argument(
        "--exts",
        nargs="+",
        default=["png", "jpg", "jpeg", "webp"],
        help="Image extensions to search for.",
    )
    parser.add_argument(
        "--prompt", type=str, default="", help="Default prompt for all images."
    )
    parser.add_argument(
        "--use_filename_as_prompt",
        action="store_true",
        help="Use the filename (without extension) as the prompt.",
    )

    args = parser.parse_args()

    data = []

    # Expected structure: root/Type/LQ/img.png and root/Type/GT/img.png
    # Iterate over all subdirectories in root
    for type_name in os.listdir(args.dataset_root):
        type_dir = os.path.join(args.dataset_root, type_name)
        if not os.path.isdir(type_dir):
            continue

        lq_dir = os.path.join(type_dir, "LQ")
        gt_dir = os.path.join(type_dir, "GT")

        if not (os.path.exists(lq_dir) and os.path.exists(gt_dir)):
            print(f"Skipping {type_name}: missing LQ or GT folder")
            continue

        print(f"Processing {type_name}...")

        lq_files = []
        for ext in args.exts:
            lq_files.extend(glob(os.path.join(lq_dir, f"*.{ext}")))

        for lq_path in lq_files:
            filename = os.path.basename(lq_path)
            gt_path = os.path.join(gt_dir, filename)

            prompt = args.prompt
            if args.use_filename_as_prompt:
                prompt = os.path.splitext(filename)[0].replace("_", " ")

            # Check exact match first
            if os.path.exists(gt_path):
                data.append(
                    {
                        "lq_path": os.path.abspath(lq_path),
                        "gt_path": os.path.abspath(gt_path),
                        "prompt": prompt,
                    }
                )
            else:
                # Try other extensions
                found = False
                base_name = os.path.splitext(filename)[0]
                for ext in args.exts:
                    gt_path_prob = os.path.join(gt_dir, f"{base_name}.{ext}")
                    if os.path.exists(gt_path_prob):
                        data.append(
                            {
                                "lq_path": os.path.abspath(lq_path),
                                "gt_path": os.path.abspath(gt_path_prob),
                                "prompt": prompt,
                            }
                        )
                        found = True
                        break
                if not found:
                    pass  # Silently skip or warn?

    print(f"Matched {len(data)} pairs total.")

    if len(data) == 0:
        print("No pairs found. Exiting.")
        return

    df = pl.DataFrame(data)
    df.write_parquet(args.output)
    print(f"Saved metadata to {args.output}")


if __name__ == "__main__":
    main()
