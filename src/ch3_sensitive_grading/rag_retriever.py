"""
检索增强上下文构建器（RAG）

算法 alg:rag-context 的完整实现。
基于 FAISS IVF-Flat 向量库 + Top-p 注意力筛选 + Few-Shot 格式化。

四个阶段：
1. 语义编码与相似度检索（Sentence-BERT + FAISS ANN）
2. Top-p 注意力筛选（温度缩放 Softmax + 累积概率截断）
3. Few-Shot 示例格式化
4. 结构化提示构建

论文对应：
    - 算法 alg:rag-context
    - 公式 (eq:sparse-attention), (eq:context-fusion)
    - FAISS IVF-Flat 配置：n_list=64, n_probe=8, 维度768
"""

import hashlib
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .prompt_builder import CorrectionCase


@dataclass
class CorrectionRecord:
    """修正案例记录（向量库条目）"""
    doc_id: str                    # 文档内容哈希
    vector: np.ndarray             # 768维 Sentence-BERT 向量
    original_level: str            # 系统原始分级
    corrected_level: str           # 管理员修正分级
    correction_reason: str         # 修正理由
    doc_summary: str               # 文档摘要
    timestamp: float = field(default_factory=time.time)


class RAGRetriever:
    """检索增强上下文构建器

    Args:
        vector_dim: 向量维度（768）
        n_list: IVF 分区数（64）
        n_probe: 搜索探测数（8）
        k: 检索候选数（默认 3）
        top_p: Top-p 筛选阈值（0.85）
        temperature: Softmax 温度参数 τ
        encode_fn: 外部编码函数 text → ndarray (vector_dim,)。
                   若提供则复用外部编码器（如 DocumentParser 的 SBERT），
                   避免重复加载模型。
    """

    def __init__(
        self,
        vector_dim: int = 768,
        n_list: int = 64,
        n_probe: int = 8,
        k: int = 3,
        top_p: float = 0.85,
        temperature: float = 1.0,
        encode_fn: Callable[[str], np.ndarray] = None,
    ):
        self.vector_dim = vector_dim
        self.n_list = n_list
        self.n_probe = n_probe
        self.k = k
        self.top_p = top_p
        self.temperature = temperature
        self._encode_fn = encode_fn

        # FAISS 索引
        self._index = None
        self._records: List[CorrectionRecord] = []
        self._sbert = None
        self._sbert_proj = None

    def _init_faiss_index(self):
        """初始化 FAISS IVF-Flat 索引"""
        try:
            import faiss
            quantizer = faiss.IndexFlatIP(self.vector_dim)  # 余弦相似度（内积）
            self._index = faiss.IndexIVFFlat(
                quantizer, self.vector_dim, self.n_list, faiss.METRIC_INNER_PRODUCT
            )
        except ImportError:
            # Fallback: 使用简单的暴力搜索
            self._index = None

    def _load_sbert(self):
        """延迟加载 Sentence-BERT，含 384→768 投影"""
        if self._sbert is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._sbert = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2"
                )
                native_dim = self._sbert.get_sentence_embedding_dimension()
                if native_dim < self.vector_dim:
                    self._sbert_proj = nn.Linear(native_dim, self.vector_dim, bias=False)
                    nn.init.xavier_uniform_(self._sbert_proj.weight)
                    self._sbert_proj.eval()
            except ImportError:
                pass

    @torch.no_grad()
    def encode_document(self, text: str) -> np.ndarray:
        """编码文档文本为 768d 向量

        优先使用外部编码器（与 DocumentParser 共享），
        否则使用内置 SBERT + 投影层。
        """
        if self._encode_fn is not None:
            return self._encode_fn(text)

        self._load_sbert()
        if self._sbert is not None:
            embedding = self._sbert.encode(text, normalize_embeddings=True)
            if self._sbert_proj is not None:
                t = torch.from_numpy(embedding).float()
                embedding = self._sbert_proj(t).numpy()
            return embedding
        return np.random.randn(self.vector_dim).astype(np.float32)

    def add_correction(self, record: CorrectionRecord):
        """添加修正案例到向量库

        冲突解决策略：版本覆盖 — 相同文档哈希仅保留最新修正。
        """
        # 检查是否存在相同文档
        for i, existing in enumerate(self._records):
            if existing.doc_id == record.doc_id:
                self._records[i] = record
                self._rebuild_index()
                return

        self._records.append(record)
        self._rebuild_index()

    def _rebuild_index(self):
        """重建 FAISS 索引"""
        if not self._records:
            return
        try:
            import faiss
            vectors = np.stack([r.vector for r in self._records]).astype(np.float32)
            faiss.normalize_L2(vectors)

            quantizer = faiss.IndexFlatIP(self.vector_dim)
            n_list = min(self.n_list, len(self._records))
            self._index = faiss.IndexIVFFlat(
                quantizer, self.vector_dim, max(1, n_list), faiss.METRIC_INNER_PRODUCT
            )
            self._index.train(vectors)
            self._index.add(vectors)
            self._index.nprobe = self.n_probe
        except ImportError:
            self._index = None

    def inject_seed_cases(self, seed_cases: List[Dict]):
        """注入种子案例（冷启动）

        每个敏感等级 5-10 个典型文档，总计 20-40 个。
        """
        for case in seed_cases:
            vector = self.encode_document(case["doc_summary"])
            record = CorrectionRecord(
                doc_id=hashlib.sha256(case["doc_summary"].encode()).hexdigest(),
                vector=vector,
                original_level=case.get("original_level", case["level"]),
                corrected_level=case["level"],
                correction_reason=case.get("reason", "种子案例"),
                doc_summary=case["doc_summary"],
            )
            self._records.append(record)
        self._rebuild_index()

    def retrieve(self, query_vector: np.ndarray, k: int = None) -> List[Tuple[CorrectionRecord, float]]:
        """阶段1: 语义编码与相似度检索

        使用 FAISS ANN 检索 k 个最近邻。

        Args:
            query_vector: 查询向量 (768维)
            k: 检索候选数

        Returns:
            [(记录, 余弦相似度), ...]
        """
        if k is None:
            k = self.k

        if not self._records:
            return []

        if self._index is not None:
            try:
                import faiss
                query = query_vector.reshape(1, -1).astype(np.float32)
                faiss.normalize_L2(query)
                scores, indices = self._index.search(query, min(k, len(self._records)))
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx >= 0 and idx < len(self._records):
                        results.append((self._records[idx], float(score)))
                return results
            except Exception:
                pass

        # Fallback: 暴力搜索
        similarities = []
        for record in self._records:
            sim = np.dot(query_vector, record.vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(record.vector) + 1e-8
            )
            similarities.append((record, float(sim)))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def top_p_filter(
        self, candidates: List[Tuple[CorrectionRecord, float]]
    ) -> List[CorrectionCase]:
        """阶段2: Top-p 注意力筛选

        1. 温度缩放 Softmax：α_i = exp(s_i/τ) / Σ exp(s_j/τ)
        2. 按 α_i 降序排列
        3. 累积概率达到 p 时截断
        4. 归一化：α̃_i = α_i / Σ_{j∈S_p} α_j

        Args:
            candidates: 检索候选列表 [(record, similarity), ...]

        Returns:
            Top-p 筛选后的修正案例列表
        """
        if not candidates:
            return []

        similarities = np.array([s for _, s in candidates])

        # 温度缩放 Softmax
        scaled = similarities / (self.temperature + 1e-8)
        scaled -= scaled.max()  # 数值稳定
        exp_scaled = np.exp(scaled)
        attention_weights = exp_scaled / (exp_scaled.sum() + 1e-8)

        # 按权重降序排序
        sorted_indices = np.argsort(attention_weights)[::-1]

        # Top-p 截断
        cumsum = 0.0
        selected_indices = []
        for idx in sorted_indices:
            cumsum += attention_weights[idx]
            selected_indices.append(idx)
            if cumsum >= self.top_p:
                break

        # 归一化权重
        selected_weights = attention_weights[selected_indices]
        normalized_weights = selected_weights / (selected_weights.sum() + 1e-8)

        # 构建结果
        results = []
        for i, idx in enumerate(selected_indices):
            record = candidates[idx][0]
            results.append(CorrectionCase(
                doc_summary=record.doc_summary,
                original_level=record.original_level,
                corrected_level=record.corrected_level,
                correction_reason=record.correction_reason,
                attention_weight=float(normalized_weights[i]),
            ))

        return results

    def build_context(
        self, document_text: str
    ) -> Tuple[List[CorrectionCase], np.ndarray]:
        """完整 RAG 上下文构建流程（算法 alg:rag-context 阶段1-2）

        Args:
            document_text: 待分级文档文本

        Returns:
            (筛选后案例列表, 查询向量)
        """
        # 阶段1: 编码与检索
        query_vector = self.encode_document(document_text)
        candidates = self.retrieve(query_vector)

        # 阶段2: Top-p 筛选
        filtered_cases = self.top_p_filter(candidates)

        return filtered_cases, query_vector

    def compute_context_fusion_vector(
        self, cases: List[CorrectionCase]
    ) -> np.ndarray:
        """计算上下文融合向量

        c_H = Σ_{i∈S_p} α̃_i · h_i

        Args:
            cases: Top-p 筛选后的修正案例

        Returns:
            融合向量 (768维)
        """
        if not cases:
            return np.zeros(self.vector_dim)

        vectors = []
        weights = []
        for case in cases:
            vec = self.encode_document(case.doc_summary)
            vectors.append(vec)
            weights.append(case.attention_weight)

        vectors = np.stack(vectors)
        weights = np.array(weights)
        fusion = np.sum(weights[:, np.newaxis] * vectors, axis=0)
        return fusion

    def process_human_feedback(
        self,
        document_text: str,
        original_level: str,
        corrected_level: str,
        correction_reason: str,
    ):
        """处理人工反馈（闭环优化）

        错误 → 修正 → 编码入库 → 检索 → 纠正

        Args:
            document_text: 被修正的文档文本
            original_level: 原始系统分级
            corrected_level: 管理员修正分级
            correction_reason: 修正理由
        """
        vector = self.encode_document(document_text)
        doc_id = hashlib.sha256(document_text.encode()).hexdigest()

        record = CorrectionRecord(
            doc_id=doc_id,
            vector=vector,
            original_level=original_level,
            corrected_level=corrected_level,
            correction_reason=correction_reason,
            doc_summary=document_text[:200],
        )
        self.add_correction(record)
