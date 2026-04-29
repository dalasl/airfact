# src/models/__init__.py
"""
模型封装层

提供统一的模型加载与推理接口，解耦各服务模块对底层框架的直接依赖。

- UNetModel: U-Net 文档去噪推理
- LayoutLMv3Wrapper: LayoutLMv3 多模态编码
- SBERTWrapper: Sentence-BERT 语义编码
- QwenWrapper: Qwen2.5-7B 敏感分级与规则解析 (Phase 3)
"""

from .unet_model import UNetModel
from .layoutlmv3_wrapper import LayoutLMv3Wrapper
from .sbert_wrapper import SBERTWrapper
from .qwen_wrapper import QwenWrapper

__all__ = [
    "UNetModel",
    "LayoutLMv3Wrapper",
    "SBERTWrapper",
    "QwenWrapper",
]
