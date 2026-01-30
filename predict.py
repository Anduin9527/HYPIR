# Batch prediction script for HYPIR
# Process all images in a folder with denoise prompt

import argparse
import os
import random
from pathlib import Path
from time import time

from accelerate.utils import set_seed
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

from HYPIR.enhancer.sd2 import SD2Enhancer


def parse_args():
    parser = argparse.ArgumentParser(description="Batch prediction script for HYPIR")
    parser.add_argument("--config", type=str, default="configs/sd2_gradio.yaml",
                        help="Path to config file")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing input images. Support nested directories.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the results.")
    parser.add_argument("--prompt", type=str, default="denoise, clean the spot",
                        help="Prompt to use for enhancement (default: 'denoise')")
    parser.add_argument("--upscale", type=int, default=1,
                        help="Upscaling factor (default: 1)")
    parser.add_argument("--patch_size", type=int, default=512,
                        help="Size of the patches to process (default: 512)")
    parser.add_argument("--stride", type=int, default=256,
                        help="Stride for the patches (default: 256)")
    parser.add_argument("--seed", type=int, default=-1,
                        help="Random seed. If -1, use random seed (default: -1)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run the model on (e.g., 'cuda', 'cpu')")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    
    # Set random seed
    if args.seed == -1:
        args.seed = random.randint(0, 2**32 - 1)
    set_seed(args.seed)
    print(f"Using seed: {args.seed}")

    # Load config
    config = OmegaConf.load(args.config)
    if config.base_model_type != "sd2":
        raise ValueError(f"Unsupported model type: {config.base_model_type}")

    # Initialize model
    print("Initializing model...")
    model = SD2Enhancer(
        base_model_path=config.base_model_path,
        weight_path=config.weight_path,
        lora_modules=config.lora_modules,
        lora_rank=config.lora_rank,
        model_t=config.model_t,
        coeff_t=config.coeff_t,
        device=args.device,
    )
    print("Loading models...")
    load_start = time()
    model.init_models()
    print(f"Models loaded in {time() - load_start:.2f} seconds.")

    # Find all images
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    images = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                full_path = Path(root) / file
                images.append(full_path)
    images.sort(key=lambda x: str(x.relative_to(input_dir)))
    print(f"Found {len(images)} images in {input_dir}.")

    if len(images) == 0:
        print("No images found! Exiting.")
        exit(1)

    # Process images
    to_tensor = transforms.ToTensor()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using prompt: '{args.prompt}'")
    print(f"Processing {len(images)} images...\n")

    for idx, file_path in enumerate(images, 1):
        print(f"[{idx}/{len(images)}] Processing: {file_path.name}")
        
        relative_path = file_path.relative_to(input_dir)
        result_path = output_dir / relative_path.with_suffix(".png")
        result_path.parent.mkdir(parents=True, exist_ok=True)

        # Load and process image
        try:
            lq_pil = Image.open(file_path).convert("RGB")
            lq_tensor = to_tensor(lq_pil).unsqueeze(0)

            process_start = time()
            result = model.enhance(
                lq=lq_tensor,
                prompt=args.prompt,
                upscale=args.upscale,
                patch_size=args.patch_size,
                stride=args.stride,
                return_type="pil",
            )[0]
            result.save(result_path)
            print(f"  ✓ Saved to {result_path} (took {time() - process_start:.2f}s)\n")
        except Exception as e:
            print(f"  ✗ Error processing {file_path.name}: {e}\n")
            continue

    print(f"Done! Results saved to {output_dir}")
