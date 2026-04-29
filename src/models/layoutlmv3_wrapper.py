"""
LayoutLMv3 多模态文档编码封装

提供统一的 LayoutLMv3 加载与推理接口，支持延迟加载和显存管理。
底层依赖: transformers (HuggingFace)
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import torch

_DEFAULT_MODEL = "microsoft/layoutlmv3-base"


class LayoutLMv3Wrapper:
    """LayoutLMv3 多模态编码器

    融合文本 token、二维布局坐标与视觉 patch，输出 768 维文档级表示。

    Args:
        model_name: HuggingFace 模型名称或本地路径
        max_seq_length: 最大序列长度
        device: 计算设备
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        max_seq_length: int = 512,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.device = device if torch.cuda.is_available() else "cpu"
        self._model = None
        self._processor = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self):
        """加载模型到指定设备"""
        if self._model is not None:
            return
        from transformers import LayoutLMv3Model, LayoutLMv3Processor
        self._processor = LayoutLMv3Processor.from_pretrained(self.model_name)
        self._model = LayoutLMv3Model.from_pretrained(self.model_name)
        self._model = self._model.to(self.device).eval()

    def unload(self):
        """释放模型显存"""
        if self._model is not None:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @torch.no_grad()
    def encode(
        self,
        image: np.ndarray,
        text: str = "",
    ) -> np.ndarray:
        """编码文档图像为 768 维向量

        Args:
            image: 文档图像 (H, W, C), uint8
            text: OCR 提取的文本（可选）

        Returns:
            768 维文档级表示向量
        """
        self.load()
        from PIL import Image as PILImage

        if isinstance(image, np.ndarray):
            pil_img = PILImage.fromarray(image)
        else:
            pil_img = image

        encoding = self._processor(
            pil_img, text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        outputs = self._model(**encoding)
        cls_vector = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        return cls_vector

    @torch.no_grad()
    def encode_batch(
        self,
        images: List[np.ndarray],
        texts: Optional[List[str]] = None,
    ) -> np.ndarray:
        """批量编码

        Args:
            images: 图像列表
            texts: 文本列表

        Returns:
            (N, 768) 向量矩阵
        """
        if texts is None:
            texts = [""] * len(images)
        vectors = []
        for img, txt in zip(images, texts):
            vectors.append(self.encode(img, txt))
        return np.stack(vectors)

    @property
    def output_dim(self) -> int:
        return 768
