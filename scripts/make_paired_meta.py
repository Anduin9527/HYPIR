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
        "--lq_dir",
        type=str,
        required=True,
        help="Path to the directory containing low-quality images.",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        required=True,
        help="Path to the directory containing ground-truth images.",
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

    lq_files = []
    for ext in args.exts:
        lq_files.extend(glob(os.path.join(args.lq_dir, f"*.{ext}")))

    print(f"Found {len(lq_files)} LQ images.")

    data = []
    for lq_path in tqdm(lq_files):
        filename = os.path.basename(lq_path)
        gt_path = os.path.join(args.gt_dir, filename)

        if os.path.exists(gt_path):
            prompt = args.prompt
            if args.use_filename_as_prompt:
                # simple clean up: replace _ with space
                prompt = os.path.splitext(filename)[0].replace("_", " ")

            data.append(
                {
                    "lq_path": os.path.abspath(lq_path),
                    "gt_path": os.path.abspath(gt_path),
                    "prompt": prompt,
                }
            )
        else:
            # Try with other extensions
            found = False
            base_name = os.path.splitext(filename)[0]
            for ext in args.exts:
                gt_path_prob = os.path.join(args.gt_dir, f"{base_name}.{ext}")
                if os.path.exists(gt_path_prob):
                    prompt = args.prompt
                    if args.use_filename_as_prompt:
                        prompt = os.path.splitext(filename)[0].replace("_", " ")

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
                print(f"Warning: GT image not found for {lq_path}")

    print(f"Matched {len(data)} pairs.")

    if len(data) == 0:
        print("No pairs found. Exiting.")
        return

    df = pl.DataFrame(data)
    df.write_parquet(args.output)
    print(f"Saved metadata to {args.output}")


if __name__ == "__main__":
    main()
