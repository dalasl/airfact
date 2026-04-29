"""
噪声注入与数据增强模块

为 U-Net 去噪模型训练生成干净-噪声配对数据。
支持的噪声类型：
- 高斯噪声（模拟扫描仪传感器噪声）
- 椒盐噪声（模拟传输损坏）
- JPEG 压缩伪影
- 文字水印叠加

输入: 干净文档图像
输出: (干净, 噪声) 配对数据
"""

import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np


def add_gaussian_noise(
    image: np.ndarray,
    sigma_range: Tuple[float, float] = (10.0, 50.0),
    seed: Optional[int] = None,
) -> np.ndarray:
    """添加高斯噪声

    Args:
        image: 输入图像 (H, W) 或 (H, W, C)，uint8
        sigma_range: 噪声标准差范围
        seed: 随机种子

    Returns:
        添加噪声后的图像，uint8
    """
    rng = np.random.RandomState(seed)
    sigma = rng.uniform(*sigma_range)
    noise = rng.normal(0, sigma, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def add_salt_pepper_noise(
    image: np.ndarray,
    density_range: Tuple[float, float] = (0.01, 0.05),
    seed: Optional[int] = None,
) -> np.ndarray:
    """添加椒盐噪声

    Args:
        image: 输入图像，uint8
        density_range: 噪声密度范围
        seed: 随机种子

    Returns:
        添加噪声后的图像
    """
    rng = np.random.RandomState(seed)
    density = rng.uniform(*density_range)
    noisy = image.copy()
    n_pixels = image.shape[0] * image.shape[1]
    n_salt = int(n_pixels * density / 2)
    n_pepper = int(n_pixels * density / 2)

    # 盐噪声（白点）
    coords_y = rng.randint(0, image.shape[0], n_salt)
    coords_x = rng.randint(0, image.shape[1], n_salt)
    noisy[coords_y, coords_x] = 255

    # 椒噪声（黑点）
    coords_y = rng.randint(0, image.shape[0], n_pepper)
    coords_x = rng.randint(0, image.shape[1], n_pepper)
    noisy[coords_y, coords_x] = 0

    return noisy


def add_jpeg_artifact(
    image: np.ndarray,
    quality_range: Tuple[int, int] = (10, 40),
    seed: Optional[int] = None,
) -> np.ndarray:
    """添加 JPEG 压缩伪影

    Args:
        image: 输入图像，uint8
        quality_range: JPEG 质量范围（越低伪影越严重）
        seed: 随机种子

    Returns:
        带压缩伪影的图像
    """
    try:
        from PIL import Image
        import io

        rng = np.random.RandomState(seed)
        quality = rng.randint(*quality_range)

        if len(image.shape) == 2:
            pil_img = Image.fromarray(image, mode="L")
        else:
            pil_img = Image.fromarray(image)

        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        compressed = Image.open(buf)
        return np.array(compressed)
    except ImportError:
        # 回退：简单的块状伪影模拟
        block_size = 8
        noisy = image.copy().astype(np.float32)
        h, w = noisy.shape[:2]
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = noisy[y:y+block_size, x:x+block_size]
                mean_val = block.mean()
                noisy[y:y+block_size, x:x+block_size] = (
                    block * 0.7 + mean_val * 0.3
                )
        return np.clip(noisy, 0, 255).astype(np.uint8)


def add_text_watermark(
    image: np.ndarray,
    text: str = "CONFIDENTIAL",
    opacity_range: Tuple[float, float] = (0.2, 0.5),
    seed: Optional[int] = None,
) -> np.ndarray:
    """添加文字水印

    Args:
        image: 输入图像，uint8
        text: 水印文字
        opacity_range: 透明度范围
        seed: 随机种子

    Returns:
        带水印的图像
    """
    try:
        from PIL import Image, ImageDraw, ImageFont

        rng = np.random.RandomState(seed)
        opacity = rng.uniform(*opacity_range)

        if len(image.shape) == 2:
            pil_img = Image.fromarray(image, mode="L").convert("RGBA")
        else:
            pil_img = Image.fromarray(image).convert("RGBA")

        # 创建水印层
        watermark = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark)

        # 字体大小自适应图像尺寸
        font_size = max(20, min(pil_img.size) // 10)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

        # 对角线重复铺设水印
        alpha = int(255 * opacity)
        step_x = font_size * len(text) // 2 + 50
        step_y = font_size * 3

        for y in range(-pil_img.size[1], pil_img.size[1] * 2, step_y):
            for x in range(-pil_img.size[0], pil_img.size[0] * 2, step_x):
                # 旋转绘制
                draw.text(
                    (x, y), text,
                    fill=(128, 128, 128, alpha),
                    font=font,
                )

        # 旋转水印层
        watermark = watermark.rotate(45, expand=False, center=(
            pil_img.size[0] // 2, pil_img.size[1] // 2,
        ))

        result = Image.alpha_composite(pil_img, watermark)
        result = result.convert("RGB")
        return np.array(result)

    except ImportError:
        # 回退：简单的条纹水印
        rng = np.random.RandomState(seed)
        opacity = rng.uniform(*opacity_range)
        noisy = image.astype(np.float32)
        h, w = noisy.shape[:2]
        stripe = np.zeros_like(noisy)
        for y in range(0, h, 80):
            stripe[y:min(y+2, h), :] = 200
        noisy = noisy * (1 - opacity) + stripe * opacity
        return np.clip(noisy, 0, 255).astype(np.uint8)


# 噪声类型注册表
NOISE_TYPES = {
    "gaussian": add_gaussian_noise,
    "salt_pepper": add_salt_pepper_noise,
    "jpeg": add_jpeg_artifact,
    "watermark": add_text_watermark,
}


def augment_single(
    image: np.ndarray,
    noise_type: Optional[str] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> Tuple[np.ndarray, str]:
    """对单张图像应用随机噪声增强

    Args:
        image: 干净图像
        noise_type: 指定噪声类型（None 则随机选择）
        seed: 随机种子

    Returns:
        (噪声图像, 噪声类型名)
    """
    rng = random.Random(seed)
    if noise_type is None:
        noise_type = rng.choice(list(NOISE_TYPES.keys()))

    fn = NOISE_TYPES[noise_type]
    noisy = fn(image, seed=seed, **kwargs)
    return noisy, noise_type


def generate_training_pairs(
    clean_dir: str,
    output_dir: str,
    noise_types: Optional[List[str]] = None,
    augments_per_image: int = 3,
    seed: int = 42,
) -> str:
    """批量生成 U-Net 训练用的干净-噪声配对

    Args:
        clean_dir: 干净图像目录
        output_dir: 输出目录（含 clean/ 和 noisy/ 子目录）
        noise_types: 使用的噪声类型列表
        augments_per_image: 每张图像生成的增强数
        seed: 随机种子

    Returns:
        配对描述文件路径
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("需要 Pillow: pip install Pillow")

    random.seed(seed)
    np.random.seed(seed)

    if noise_types is None:
        noise_types = list(NOISE_TYPES.keys())

    clean_out = os.path.join(output_dir, "clean")
    noisy_out = os.path.join(output_dir, "noisy")
    os.makedirs(clean_out, exist_ok=True)
    os.makedirs(noisy_out, exist_ok=True)

    pairs_file = os.path.join(output_dir, "pairs.csv")
    import csv

    pair_count = 0
    with open(pairs_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "clean_file", "noisy_file", "noise_type", "source_file"])

        supported_ext = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
        for fname in sorted(os.listdir(clean_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in supported_ext:
                continue

            img_path = os.path.join(clean_dir, fname)
            try:
                pil_img = Image.open(img_path)
                img_array = np.array(pil_img)
            except Exception as e:
                print(f"[WARN] 无法读取 {fname}: {e}")
                continue

            # 保存干净版本
            stem = os.path.splitext(fname)[0]
            clean_fname = f"{stem}.png"
            clean_path = os.path.join(clean_out, clean_fname)
            Image.fromarray(img_array).save(clean_path)

            # 生成多个噪声版本
            for aug_i in range(augments_per_image):
                nt = noise_types[aug_i % len(noise_types)]
                aug_seed = seed + pair_count
                noisy_img, actual_nt = augment_single(
                    img_array, noise_type=nt, seed=aug_seed,
                )
                noisy_fname = f"{stem}_aug{aug_i}_{actual_nt}.png"
                noisy_path = os.path.join(noisy_out, noisy_fname)
                Image.fromarray(noisy_img).save(noisy_path)

                writer.writerow([
                    f"pair_{pair_count:06d}",
                    clean_fname,
                    noisy_fname,
                    actual_nt,
                    fname,
                ])
                pair_count += 1

    print(f"[噪声注入] 完成: {pair_count} 个配对")
    print(f"  干净图像: {clean_out}")
    print(f"  噪声图像: {noisy_out}")
    print(f"  配对文件: {pairs_file}")
    return pairs_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="噪声注入与训练对生成")
    parser.add_argument("--clean-dir", required=True, help="干净图像目录")
    parser.add_argument(
        "--output-dir", default="data/unet_pairs", help="输出目录",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augments", type=int, default=3, help="每张图的增强数")
    args = parser.parse_args()

    generate_training_pairs(
        clean_dir=args.clean_dir,
        output_dir=args.output_dir,
        augments_per_image=args.augments,
        seed=args.seed,
    )
