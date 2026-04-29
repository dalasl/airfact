# src/ch3_sensitive_grading/__init__.py
"""
第3章：基于上下文增强的敏感数据智能分级方法

模块功能：
- 五层结构化提示工程（画像、经验、数据三维增强）
- 基于 Top-p 注意力筛选的检索增强生成（FAISS + Sentence-BERT）
- 不确定性量化与自洽性校验（MC-Dropout + 多数投票）
- 敏感资产目录的动态维护
- 端到端分级流水线
"""

from .prompt_builder import PromptBuilder
from .rag_retriever import RAGRetriever
from .self_consistency import SelfConsistencyVerifier
from .asset_catalog import SensitiveAssetCatalog
from .grading_pipeline import GradingPipeline

__all__ = [
    "PromptBuilder",
    "RAGRetriever",
    "SelfConsistencyVerifier",
    "SensitiveAssetCatalog",
    "GradingPipeline",
]
