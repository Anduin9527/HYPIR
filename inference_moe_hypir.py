"""
简化的推理脚本，用于配合 NAFNetMOE 一体化前端
使用训练好的 HYPIR 模型进行最终图像生成
"""

import argparse
import os
from pathlib import Path
from time import time
import torch

from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms

from HYPIR.enhancer.sd2 import SD2Enhancer
from HYPIR.utils.captioner import EmptyCaptioner, FixedCaptioner

# 导入 NAFNetMOE
from HYPIR.model.nafnet_moe_wrapper import NAFNetMOEWrapper


def parse_args():
    parser = argparse.ArgumentParser(description="HYPIR + NAFNetMOE 联合推理脚本")

    # 必需参数
    parser.add_argument(
        "--lq_dir", type=str, required=True, help="输入的低质量图像文件夹路径"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="输出结果文件夹路径"
    )

    # HYPIR 参数
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./results/checkpoint-50000",
        help="HYPIR checkpoint 路径",
    )
    parser.add_argument(
        "--base_model_path", type=str, default="stabilityai/stable-diffusion-2-1-base"
    )
    parser.add_argument("--model_t", type=int, default=200)
    parser.add_argument("--coeff_t", type=int, default=200)
    parser.add_argument("--lora_rank", type=int, default=256)
    parser.add_argument(
        "--lora_modules",
        type=str,
        default="to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj",
    )

    # NAFNetMOE 参数
    parser.add_argument(
        "--moe_checkpoint",
        type=str,
        required=True,
        help="NAFNet MOE Router (或者完整 MOE) 的权重路径",
    )
    parser.add_argument("--moe_num_experts", type=int, default=5, help="MOE 专家数量")
    parser.add_argument(
        "--moe_top_k",
        type=int,
        default=2,
        help="推理时激活的 Top-K 专家数量 (节省显存和计算)，传 5 为全激活",
    )

    # 推理参数
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--upscale", type=int, default=1)

    # Prompt 设置
    parser.add_argument("--prompt", type=str, default="high quality, sharp details")
    parser.add_argument("--txt_dir", type=str, default=None)

    # 其他参数
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 60)
    print("1. 加载 NAFNet+MOE 前端条件提取模块...")

    # NAFNetMOE 初始化
    # 读取 checkpoint 里面的 config 可以自动获得网络超参，没找到则用默认
    moe_state = torch.load(args.moe_checkpoint, map_location="cpu")
    moe_config = moe_state.get("config", {}).get("nafnet", {})

    nafnet_moe = NAFNetMOEWrapper(
        checkpoint_path=args.moe_checkpoint,
        num_experts=args.moe_num_experts,
        width=moe_config.get("width", 64),
        enc_blks=moe_config.get("enc_blk_nums", [2, 2, 4, 8]),
        middle_blk_num=moe_config.get("middle_blk_num", 12),
        dec_blks=moe_config.get("dec_blk_nums", [2, 2, 2, 2]),
        freeze=True,
    )
    nafnet_moe.eval().to(args.device)
    print(f"  - NAFNet+MOE 成功加载，推理模式配置 Top-K = {args.moe_top_k}")

    print("=" * 60)
    print("2. 加载 HYPIR 生成式模块...")

    checkpoint_dir = Path(args.checkpoint)
    if args.use_ema and (checkpoint_dir / "ema_state_dict.pth").exists():
        weight_path = str(checkpoint_dir / "ema_state_dict.pth")
    elif (checkpoint_dir / "state_dict.pth").exists():
        weight_path = str(checkpoint_dir / "state_dict.pth")
    else:
        raise FileNotFoundError(f"在 {checkpoint_dir} 中未找到权重文件")

    load_start = time()
    model = SD2Enhancer(
        base_model_path=args.base_model_path,
        weight_path=weight_path,
        lora_modules=args.lora_modules.split(","),
        lora_rank=args.lora_rank,
        model_t=args.model_t,
        coeff_t=args.coeff_t,
        device=args.device,
    )
    model.init_models()
    print(f"  - HYPIR 加载完成，耗时 {time() - load_start:.2f} 秒")
    print("=" * 60)

    # 查找所有输入图像
    input_dir = Path(args.lq_dir)
    output_dir = Path(args.output_dir)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    images = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                images.append(Path(root) / file)
    images.sort(key=lambda x: str(x.relative_to(input_dir)))

    if len(images) == 0:
        print(f"错误：在 {input_dir} 中未找到任何图像文件")
        return

    # 建立输出目录结构
    result_dir = output_dir / "result"
    moe_out_dir = output_dir / "moe_pre_result"
    result_dir.mkdir(parents=True, exist_ok=True)
    moe_out_dir.mkdir(parents=True, exist_ok=True)

    to_tensor = transforms.ToTensor()

    # Captioner
    captioner = (
        FixedCaptioner(args.device, args.prompt)
        if args.prompt
        else EmptyCaptioner(args.device)
    )

    total_time = 0
    from torchvision.transforms import ToPILImage

    to_pil = ToPILImage()

    for idx, file_path in enumerate(images, 1):
        print(f"\n[{idx}/{len(images)}] 处理: {file_path.name}")

        # 准备路径
        rel_path = file_path.relative_to(input_dir)
        res_path = result_dir / rel_path.with_suffix(".jpg")
        moe_path = moe_out_dir / rel_path.with_suffix(".png")
        res_path.parent.mkdir(parents=True, exist_ok=True)
        moe_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. 图像预处理
        lq_pil = Image.open(file_path).convert("RGB")
        lq_tensor = (
            to_tensor(lq_pil).unsqueeze(0).to(args.device)
        )  # [1, 3, H, W] in [0, 1]

        # 转换到 HYPIR standard ([-1, 1]) 因为 NAFNetMOE 包装器期望这个
        lq_input_neg1_1 = lq_tensor * 2 - 1

        infer_start = time()

        # 2. MOE 提取初步清晰 Condition
        with torch.no_grad():
            # out_neg1_1 范围 [-1, 1]
            moe_condition, bal_loss, weights = nafnet_moe(
                lq_input_neg1_1, top_k=args.moe_top_k, return_routing=True
            )

            # Print routing info
            w_str = ", ".join([f"{w.item():.2f}" for w in weights[0]])
            print(f"  - MOE 专家权重分布: [{w_str}]")

            # 保存中间结果供对比
            moe_condition_01 = ((moe_condition + 1) / 2).cpu().clamp(0, 1).squeeze(0)
            moe_pil = to_pil(moe_condition_01)
            moe_pil.save(moe_path)

        # 3. 准备 Prompt
        prompt = captioner(lq_pil)

        # 4. 经过 HYPIR
        # 注意：这里我们用 moe_condition 作为 LQ 送给 Enhance
        # Enhance 内部会再次将送到 VAE 和 SD
        result = model.enhance(
            lq=moe_condition.cpu(),  # HYPIR enhance 函数期望 CPU tensor 或者 [-1, 1]
            prompt=prompt,
            scale_by="factor",
            upscale=args.upscale,
            patch_size=args.patch_size,
            stride=args.stride,
            return_type="pil",
        )[0]

        infer_time = time() - infer_start
        total_time += infer_time

        # 保存 HYPIR 结果
        if result.mode != "RGB":
            result = result.convert("RGB")
        result.save(res_path, format="JPEG", quality=96, optimize=True)

        print(f"  - 最终结果保存至: {res_path}")
        print(f"  - 耗时: {infer_time:.2f} 秒")

    print("\n" + "=" * 60)
    print(f"✅ 完成！共处理 {len(images)} 张图像")
    print(f"总耗时: {total_time:.2f} 秒，平均 {total_time / len(images):.2f} 秒/张")
    print(f"初步 Condition 结果保存在: {moe_out_dir}")
    print(f"最终 HYPIR 结果保存在: {result_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
