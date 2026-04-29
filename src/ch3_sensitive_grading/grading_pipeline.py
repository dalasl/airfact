"""
端到端分级流水线

整合五层提示构建、RAG 检索增强、LLM 推理、自洽性校验
的完整分级流程。

流程：
Document → SentenceBERT → FAISS检索 → Top-p筛选
→ 五层提示构建 → Qwen-7B (4-bit, MC-Dropout ×T)
→ 不确定性量化 → 自洽性投票 → 决策/人工审核
→ 敏感资产目录更新

论文对应：第3章完整方法链路
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .prompt_builder import PromptBuilder, UserProfile, CorrectionCase
from .rag_retriever import RAGRetriever
from .self_consistency import SelfConsistencyVerifier, GradingDecision
from .asset_catalog import SensitiveAssetCatalog


class GradingPipeline:
    """端到端敏感数据分级流水线

    Args:
        llm_fn: LLM 推理函数 (prompt, seed) → raw_text
        sbert_model: Sentence-BERT 模型名称
        config: 分级配置字典
    """

    def __init__(
        self,
        llm_fn: Callable = None,
        sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        config: Dict[str, Any] = None,
    ):
        config = config or {}

        self.llm_fn = llm_fn
        self.prompt_builder = PromptBuilder()

        # RAG 检索器
        rag_config = config.get("rag", {})
        self.retriever = RAGRetriever(
            k=rag_config.get("k", 3),
            top_p=rag_config.get("top_p", 0.85),
            temperature=rag_config.get("temperature", 1.0),
        )

        # 自洽性校验器
        sc_config = config.get("self_consistency", {})
        self.verifier = SelfConsistencyVerifier(
            T=sc_config.get("T", 5),
            rho_min=sc_config.get("rho_min", 0.6),
            c_min=sc_config.get("c_min", 0.85),
            gamma=sc_config.get("gamma", 1.5),
            beta=sc_config.get("beta", 0.6),
        )

        # 资产目录
        self.catalog = SensitiveAssetCatalog()

    def grade_document(
        self,
        file_path: str,
        content_snippet: str,
        profile: UserProfile,
        metadata: Dict[str, Any] = None,
    ) -> GradingDecision:
        """分级单个文档

        Args:
            file_path: 文件路径
            content_snippet: 文件内容片段
            profile: 用户画像
            metadata: 附加元数据

        Returns:
            分级决策
        """
        # 步骤1: RAG 上下文构建
        cases, query_vector = self.retriever.build_context(content_snippet)

        # 步骤2: 五层提示构建
        prompt = self.prompt_builder.build_prompt(
            profile=profile,
            cases=cases,
            file_path=file_path,
            content_snippet=content_snippet,
            metadata=metadata,
        )

        # 步骤3: 自洽性校验（含 MC-Dropout + 多数投票）
        decision = self.verifier.verify(self.llm_fn, prompt)

        return decision

    def grade_batch(
        self,
        documents: List[Dict[str, Any]],
        profile: UserProfile,
    ) -> List[GradingDecision]:
        """批量分级

        Args:
            documents: 文档列表 [{file_path, content_snippet, metadata}, ...]
            profile: 用户画像

        Returns:
            分级决策列表
        """
        results = []
        for doc in documents:
            decision = self.grade_document(
                file_path=doc["file_path"],
                content_snippet=doc["content_snippet"],
                profile=profile,
                metadata=doc.get("metadata"),
            )
            results.append(decision)
        return results

    def run_ablation_no_profile(
        self, file_path: str, content_snippet: str
    ) -> GradingDecision:
        """消融实验：无画像注入"""
        prompt = self.prompt_builder.build_zero_shot_prompt(file_path, content_snippet)
        return self.verifier.verify(self.llm_fn, prompt)

    def run_ablation_no_rag(
        self, file_path: str, content_snippet: str, profile: UserProfile
    ) -> GradingDecision:
        """消融实验：无 RAG"""
        prompt = self.prompt_builder.build_prompt(
            profile=profile, cases=[], file_path=file_path,
            content_snippet=content_snippet,
        )
        return self.verifier.verify(self.llm_fn, prompt)

    def run_ablation_no_consistency(
        self, file_path: str, content_snippet: str, profile: UserProfile
    ) -> GradingDecision:
        """消融实验：无自洽性校验（T=1）"""
        cases, _ = self.retriever.build_context(content_snippet)
        prompt = self.prompt_builder.build_prompt(
            profile=profile, cases=cases, file_path=file_path,
            content_snippet=content_snippet,
        )

        # 单次推理
        raw = self.llm_fn(prompt, seed=0)
        result = self.verifier._parse_llm_output(raw)
        return GradingDecision(
            level=result.level,
            confidence=result.confidence,
            consistency_ratio=1.0,
            uncertainty=0.0,
            is_accepted=True,
            needs_human_review=False,
            reason=result.reason,
            all_predictions=[result.level],
        )
