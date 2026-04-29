"""
U-Net 去噪模型推理封装

封装 UNetDenoiser 的模型加载、权重管理与推理接口。
训练逻辑见 scripts/train_unet.py。
"""

import os
from typing import Optional

import numpy as np
import torch

from src.ch2_user_profiling.unet_denoiser import UNetDenoiser


class UNetModel:
    """U-Net 去噪模型推理接口

    Args:
        weights_path: 预训练权重路径，为 None 则使用随机初始化
        in_channels: 输入通道数（灰度=1, RGB=3）
        alpha: MSE/SSIM 损失平衡系数
        device: 计算设备
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        in_channels: int = 3,
        alpha: float = 0.7,
        device: str = "cuda",
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = UNetDenoiser(
            in_channels=in_channels,
            out_channels=in_channels,
            alpha=alpha,
        )
        if weights_path and os.path.isfile(weights_path):
            state = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
        self.model = self.model.to(self.device)
        self.model.eval()

    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """去噪单张图像

        Args:
            image: uint8 图像 (H, W, C) 或 (H, W)

        Returns:
            去噪后的 uint8 图像，形状与输入相同
        """
        is_gray = image.ndim == 2
        if is_gray:
            image = image[:, :, np.newaxis]

        # HWC -> BCHW, 归一化到 [0,1]
        tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)

        with torch.no_grad():
            output = self.model(tensor)

        # BCHW -> HWC, 反归一化
        result = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        result = np.clip(result * 255, 0, 255).astype(np.uint8)

        if is_gray:
            result = result.squeeze(-1)
        return result

    def denoise_batch(self, images: list) -> list:
        """批量去噪

        Args:
            images: uint8 图像列表

        Returns:
            去噪后的图像列表
        """
        return [self.denoise_image(img) for img in images]

    def save_weights(self, path: str):
        """保存模型权重"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
