"""
Sentence-BERT 语义编码封装

提供统一的文本语义向量编码接口，支持中英文文档。
底层依赖: sentence-transformers
"""

from typing import List, Optional, Union

import numpy as np

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_MULTILINGUAL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class SBERTWrapper:
    """Sentence-BERT 语义编码器

    将文本编码为归一化的 768 维语义向量（或模型原生维度），
    用于文档语义相似度计算和 RAG 查询。

    Args:
        model_name: 模型名称或本地路径
        device: 计算设备
        normalize: 是否 L2 归一化输出向量
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str = "cuda",
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self):
        """加载模型"""
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name, device=self.device)

    def unload(self):
        """释放模型"""
        if self._model is not None:
            del self._model
            self._model = None

    def encode(self, text: str) -> np.ndarray:
        """编码单个文本

        Args:
            text: 输入文本

        Returns:
            语义向量 (dim,)
        """
        self.load()
        return self._model.encode(text, normalize_embeddings=self.normalize)

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """批量编码

        Args:
            texts: 文本列表
            batch_size: 批大小

        Returns:
            语义向量矩阵 (N, dim)
        """
        self.load()
        return self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )

    @property
    def output_dim(self) -> int:
        self.load()
        return self._model.get_sentence_embedding_dimension()
