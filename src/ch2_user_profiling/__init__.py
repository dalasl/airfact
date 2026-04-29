# src/ch2_user_profiling/__init__.py
"""
第2章：面向终端异构数据的动态用户画像构建方法

模块功能：
- U-Net 文档抗噪模型
- LayoutLMv3 + Sentence-BERT 多模态文档解析
- 多源异构数据时空对齐
- 三维特征空间构建（内容语义 + 行为模式 + 权限上下文）
- 双支路交叉注意力融合编码器
- 信息增益加权 K-means 聚类（IGW-Kmeans）
- 事件驱动增量更新机制
"""

from .unet_denoiser import UNetDenoiser
from .document_parser import DocumentParser
from .temporal_alignment import TemporalAligner
from .feature_extraction import FeatureExtractor
from .fusion_encoder import DualBranchFusionEncoder
from .igw_kmeans import IGWKMeans
from .incremental_update import IncrementalUpdater

__all__ = [
    "UNetDenoiser",
    "DocumentParser",
    "TemporalAligner",
    "FeatureExtractor",
    "DualBranchFusionEncoder",
    "IGWKMeans",
    "IncrementalUpdater",
]
