"""
U-Net 去噪模型训练脚本

使用 noise_augmentor 生成的干净-噪声配对数据训练 U-Net。
损失函数: L = α·MSE + (1-α)·(1-SSIM), α=0.7

用法:
    python -m scripts.train_unet --data-dir data/unet_pairs --epochs 100
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from src.ch2_user_profiling.unet_denoiser import UNetDenoiser


class DenoisePairDataset(Dataset):
    """干净-噪声配对数据集

    从 noise_augmentor 生成的 pairs.csv 加载配对数据。
    """

    def __init__(self, data_dir: str, image_size: int = 256):
        self.clean_dir = os.path.join(data_dir, "clean")
        self.noisy_dir = os.path.join(data_dir, "noisy")
        self.image_size = image_size
        self.pairs = []

        pairs_csv = os.path.join(data_dir, "pairs.csv")
        if not os.path.isfile(pairs_csv):
            raise FileNotFoundError(f"pairs.csv not found at {pairs_csv}")

        with open(pairs_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                clean_path = os.path.join(self.clean_dir, row["clean_file"])
                noisy_path = os.path.join(self.noisy_dir, row["noisy_file"])
                if os.path.isfile(clean_path) and os.path.isfile(noisy_path):
                    self.pairs.append((clean_path, noisy_path))

    def __len__(self):
        return len(self.pairs)

    def _load_image(self, path: str) -> np.ndarray:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0

    def __getitem__(self, idx):
        clean_path, noisy_path = self.pairs[idx]
        clean = self._load_image(clean_path)
        noisy = self._load_image(noisy_path)
        # HWC -> CHW
        clean = torch.from_numpy(clean).permute(2, 0, 1)
        noisy = torch.from_numpy(noisy).permute(2, 0, 1)
        return noisy, clean


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[U-Net Train] Device: {device}")

    # 数据集
    dataset = DenoisePairDataset(args.data_dir, image_size=args.image_size)
    print(f"[U-Net Train] Total pairs: {len(dataset)}")
    if len(dataset) == 0:
        print("[ERROR] No training pairs found. Run noise_augmentor first.")
        return

    # 训练/验证拆分 (90/10)
    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 模型
    model = UNetDenoiser(in_channels=3, out_channels=3, alpha=args.alpha).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
    )

    # 加载检查点
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[U-Net Train] Resumed from epoch {start_epoch}")

    best_val_loss = float("inf")
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        t0 = time.time()
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            pred = model(noisy)
            loss = model.compute_loss(pred, clean)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                pred = model(noisy)
                loss = model.compute_loss(pred, clean)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:03d}/{args.epochs} | "
            f"train_loss={train_loss:.5f} | val_loss={val_loss:.5f} | "
            f"lr={lr:.2e} | {elapsed:.1f}s"
        )

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.output_dir, "unet_denoiser.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  -> Best model saved: val_loss={val_loss:.5f}")

        # 定期检查点
        if (epoch + 1) % 20 == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }, ckpt_path)

    print(f"\n[U-Net Train] Done. Best val_loss={best_val_loss:.5f}")
    print(f"  Model: {os.path.join(args.output_dir, 'unet_denoiser.pth')}")

    # 参数量统计
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U-Net 去噪模型训练")
    parser.add_argument("--data-dir", default="data/unet_pairs", help="配对数据目录")
    parser.add_argument("--output-dir", default="models", help="模型输出目录")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.7, help="MSE/SSIM 平衡系数")
    parser.add_argument("--image-size", type=int, default=256, help="训练图像尺寸")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None, help="检查点路径")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)
