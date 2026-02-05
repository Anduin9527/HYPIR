"""
简化的推理脚本，用于对 LQ 图像文件夹进行批量修复
使用训练好的 HYPIR 模型进行图像增强
"""
import argparse
import os
from pathlib import Path
from time import time

from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms

from HYPIR.enhancer.sd2 import SD2Enhancer
from HYPIR.utils.captioner import EmptyCaptioner, FixedCaptioner


def parse_args():
    parser = argparse.ArgumentParser(description="HYPIR 图像修复推理脚本")
    
    # 必需参数
    parser.add_argument("--lq_dir", type=str, required=True,
                        help="输入的低质量图像文件夹路径")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出结果文件夹路径")
    parser.add_argument("--checkpoint", type=str, default="./results/checkpoint-50000",
                        help="训练好的 checkpoint 路径 (默认: ./results/checkpoint-50000)")
    
    # 模型配置参数（应与训练配置一致）
    parser.add_argument("--base_model_path", type=str, default="stabilityai/stable-diffusion-2-1-base",
                        help="基础模型路径")
    parser.add_argument("--model_t", type=int, default=200,
                        help="模型时间步")
    parser.add_argument("--coeff_t", type=int, default=200,
                        help="系数时间步")
    parser.add_argument("--lora_rank", type=int, default=256,
                        help="LoRA rank")
    parser.add_argument("--lora_modules", type=str, 
                        default="to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj",
                        help="LoRA 模块列表")
    
    # 推理参数
    parser.add_argument("--patch_size", type=int, default=512,
                        help="处理的 patch 大小")
    parser.add_argument("--stride", type=int, default=256,
                        help="Patch 滑动步长")
    parser.add_argument("--upscale", type=int, default=1,
                        help="上采样倍数 (1=不上采样)")
    
    # Prompt 设置（重要：模型是用 prompt 训练的，推理时也应使用）
    parser.add_argument("--prompt", type=str, default="high quality, sharp details",
                        help="图像修复的 prompt（模型使用 prompt 训练，推荐使用）")
    parser.add_argument("--txt_dir", type=str, default=None,
                        help="包含每张图片对应 prompt 的文件夹（优先级高于 --prompt）")
    
    # LPIPS 评估参数（可选）
    parser.add_argument("--gt_dir", type=str, default=None,
                        help="Ground Truth 目录（用于计算 LPIPS，可选）")
    parser.add_argument("--skip_lpips", action="store_true",
                        help="跳过 LPIPS 计算")
    
    # 其他参数
    parser.add_argument("--use_ema", action="store_true", default=True,
                        help="使用 EMA 权重（推荐）")
    parser.add_argument("--seed", type=int, default=231,
                        help="随机种子")
    parser.add_argument("--device", type=str, default="cuda",
                        help="运行设备")
    
    # 级联去模糊选项
    parser.add_argument("--use_deblur", action="store_true",
                        help="使用去模糊预处理（级联模式）")
    parser.add_argument("--deblur_checkpoint", type=str, default=None,
                        help="去模糊模块 checkpoint 路径")
    parser.add_argument("--blur_threshold", type=float, default=0.5,
                        help="isBlur 分类阈值（> threshold 认为是模糊）")
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # 加载去模糊模块（如果启用）
    isblur_model = None
    nafnet_model = None
    if args.use_deblur:
        if not args.deblur_checkpoint:
            raise ValueError("启用 --use_deblur 时必须指定 --deblur_checkpoint")
        
        print("=" * 60)
        print("加载去模糊模块...")
        
        import torch
        from HYPIR.model.isblur import BlurClassifier
        from HYPIR.model.nafnet_wrapper import NAFNetWrapper
        
        checkpoint = torch.load(args.deblur_checkpoint, map_location='cpu')
        deblur_config = checkpoint.get('config', {})
        
        # 加载 isBlur
        isblur_model = BlurClassifier(
            backbone=deblur_config.get('isblur', {}).get('backbone', 'resnet18')
        )
        isblur_model.load_state_dict(checkpoint['isblur'])
        isblur_model.eval().to(args.device)
        print(f"  - isBlur 分类器加载完成")
        
        # 加载 NAFNet
        nafnet_config = deblur_config.get('nafnet', {})
        nafnet_model = NAFNetWrapper(
            checkpoint_path=None,
            width=nafnet_config.get('width', 64),
            enc_blks=nafnet_config.get('enc_blks', [2, 2, 4, 8]),
            middle_blk_num=nafnet_config.get('middle_blk_num', 12),
            dec_blks=nafnet_config.get('dec_blks', [2, 2, 2, 2]),
            freeze=True
        )
        nafnet_model.nafnet.load_state_dict(checkpoint['nafnet'])
        nafnet_model.eval().to(args.device)
        print(f"  - NAFNet 去模糊网络加载完成")
        print(f"  - 阈值: {args.blur_threshold}")
        print("=" * 60)
    
    # 确定权重文件路径
    checkpoint_dir = Path(args.checkpoint)
    if args.use_ema and (checkpoint_dir / "ema_state_dict.pth").exists():
        weight_path = str(checkpoint_dir / "ema_state_dict.pth")
        print(f"使用 EMA 权重: {weight_path}")
    elif (checkpoint_dir / "state_dict.pth").exists():
        weight_path = str(checkpoint_dir / "state_dict.pth")
        print(f"使用标准权重: {weight_path}")
    else:
        raise FileNotFoundError(f"在 {checkpoint_dir} 中未找到权重文件")
    
    # 初始化模型
    print("=" * 60)
    print("开始加载模型...")
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
    
    print(f"模型加载完成，耗时 {time() - load_start:.2f} 秒")
    print("=" * 60)
    
    # 查找所有输入图像
    input_dir = Path(args.lq_dir)
    output_dir = Path(args.output_dir)
    
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    images = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                full_path = Path(root) / file
                images.append(full_path)
    images.sort(key=lambda x: str(x.relative_to(input_dir)))
    
    if len(images) == 0:
        print(f"错误：在 {input_dir} 中未找到任何图像文件")
        return
    
    print(f"在 {input_dir} 中找到 {len(images)} 张图像")
    print("=" * 60)
    
    # 设置 captioner
    if args.txt_dir is None:
        if args.prompt:
            captioner = FixedCaptioner(args.device, args.prompt)
            print(f"使用固定 prompt: '{args.prompt}'")
        else:
            captioner = EmptyCaptioner(args.device)
            print("使用空 prompt")
    else:
        print(f"从 {args.txt_dir} 读取 prompt")
        captioner = None
    
    print("=" * 60)
    
    # 准备输出目录
    result_dir = output_dir / "result"
    prompt_dir = output_dir / "prompt"
    result_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    
    to_tensor = transforms.ToTensor()
    
    # 处理每张图像
    total_time = 0
    for idx, file_path in enumerate(images, 1):
        print(f"\n[{idx}/{len(images)}] 处理: {file_path.name}")
        
        relative_path = file_path.relative_to(input_dir)
        result_path = result_dir / relative_path.with_suffix(".png")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path = prompt_dir / relative_path.with_suffix(".txt")
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 读取图像
        lq_pil = Image.open(file_path).convert("RGB")
        print(f"  输入尺寸: {lq_pil.size}")
        lq_tensor = to_tensor(lq_pil).unsqueeze(0)
        
        # 去模糊预处理（如果启用）
        if args.use_deblur and isblur_model is not None and nafnet_model is not None:
            import torch
            lq_input = lq_tensor * 2 - 1  # [0, 1] -> [-1, 1]
            lq_input = lq_input.to(args.device)
            
            with torch.no_grad():
                # isBlur 判断
                is_blur_logits = isblur_model(lq_input)
                is_blur_prob = torch.sigmoid(is_blur_logits).item()
                
                if is_blur_prob > args.blur_threshold:
                    print(f"  检测到模糊 (概率={is_blur_prob:.3f})，应用 NAFNet 去模糊...")
                    # NAFNet 去模糊
                    deblurred = nafnet_model(lq_input)
                    # 转换回 [0, 1] 并更新 tensor
                    lq_tensor = ((deblurred + 1) / 2).cpu()
                else:
                    print(f"  未检测到模糊 (概率={is_blur_prob:.3f})，跳过去模糊")
                
                # 更新 PIL 图像（用于后续处理）
                from torchvision.transforms import ToPILImage
                to_pil = ToPILImage()
                lq_pil = to_pil(lq_tensor.squeeze(0))
        
        # 获取 prompt
        if args.txt_dir is not None:
            txt_path = Path(args.txt_dir) / relative_path.with_suffix(".txt")
            if txt_path.exists():
                with open(txt_path, "r") as fp:
                    prompt = fp.read().strip()
            else:
                print(f"  警告: 未找到 prompt 文件 {txt_path}，使用空 prompt")
                prompt = ""
        else:
            prompt = captioner(lq_pil)
        
        # 保存 prompt
        with open(prompt_path, "w") as fp:
            fp.write(prompt)
        if prompt:
            print(f"  Prompt: '{prompt}'")
        
        # 推理
        infer_start = time()
        result = model.enhance(
            lq=lq_tensor,
            prompt=prompt,
            scale_by="factor",
            upscale=args.upscale,
            patch_size=args.patch_size,
            stride=args.stride,
            return_type="pil",
        )[0]
        infer_time = time() - infer_start
        total_time += infer_time
        
        # 保存结果（使用高质量 JPEG 格式）
        if result.mode != "RGB":
            result = result.convert("RGB")
        save_kwargs = dict(format="JPEG", quality=96, subsampling=0, optimize=True)
        # 修改扩展名为 .jpg
        result_path = result_path.with_suffix(".jpg")
        result.save(result_path, **save_kwargs)
        print(f"  输出尺寸: {result.size}")
        print(f"  保存至: {result_path}")
        print(f"  耗时: {infer_time:.2f} 秒")
    
    # 总结
    print("\n" + "=" * 60)
    print(f"✅ 完成！共处理 {len(images)} 张图像")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均耗时: {total_time/len(images):.2f} 秒/张")
    print(f"结果保存在: {result_dir}")
    print("=" * 60)
    
    return result_dir  # 返回结果目录路径


if __name__ == "__main__":
    args = parse_args()
    result_dir = main()
    
    # 自动运行 LPIPS 计算（如果提供了 GT 目录且未跳过）
    if args.gt_dir and not args.skip_lpips:
        print("\n" + "=" * 60)
        print("开始计算 LPIPS 指标...")
        print("=" * 60)
        
        import subprocess
        import sys
        
        output_csv = Path(args.output_dir) / "lpips_results.csv"
        cmd = [
            sys.executable, "calculate_lpips.py",
            "--baseline_dir", str(result_dir),  # 单模型评估，baseline和finetuned都用同一个
            "--finetuned_dir", str(result_dir),
            "--gt_dir", args.gt_dir,
            "--output_csv", str(output_csv),
            "--device", args.device
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"⚠️  LPIPS 计算失败: {e}")
        except FileNotFoundError:
            print("⚠️  未找到 calculate_lpips.py，跳过 LPIPS 计算")
