"""
不确定性量化与自洽性校验

算法 alg:context-grading 的核心实现。
通过 MC-Dropout 多轮推理 + 多数投票实现输出稳定性保障。

两阶段流程：
1. MC-Dropout 不确定性量化：T=5 轮随机前向传播 → 预测均值/方差 → 不确定性评分
2. 自洽性多数投票：一致性比率 ρ + 置信度加权评分 c̄ → 双阈值决策

论文对应：
    - 算法 alg:context-grading（阶段3-5）
    - 公式 (eq:mc-mean), (eq:mc-var), (eq:uncertainty-score)
    - 公式 (eq:majority-vote), (eq:consistency-rate), (eq:confidence-weight)
"""

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class InferenceResult:
    """单次推理结果"""
    level: str          # L1/L2/L3/L4
    confidence: float   # 模型置信度
    reason: str = ""    # 判定理由
    category: str = ""  # 数据类别


@dataclass
class GradingDecision:
    """最终分级决策"""
    level: str                         # 最终分级 L1/L2/L3/L4
    confidence: float                  # 置信度加权评分
    consistency_ratio: float           # 一致性比率 ρ
    uncertainty: float                 # 不确定性评分 U(d)
    is_accepted: bool                  # 是否通过自洽性校验
    needs_human_review: bool           # 是否需要人工审核
    reason: str = ""                   # 判定理由
    all_predictions: List[str] = None  # 所有轮次预测（调试用）


class SelfConsistencyVerifier:
    """不确定性量化与自洽性校验器

    Args:
        T: MC-Dropout 推理轮次（默认 5）
        rho_min: 一致性比率阈值（0.6）
        c_min: 置信度阈值（0.85）
        gamma: 不确定性阈值（1.5）
        beta: 熵与方差平衡系数（0.6）
    """

    LEVEL_INDEX = {"L1": 0, "L2": 1, "L3": 2, "L4": 3}
    INDEX_LEVEL = {0: "L1", 1: "L2", 2: "L3", 3: "L4"}

    def __init__(
        self,
        T: int = 5,
        rho_min: float = 0.6,
        c_min: float = 0.85,
        gamma: float = 1.5,
        beta: float = 0.6,
    ):
        self.T = T
        self.rho_min = rho_min
        self.c_min = c_min
        self.gamma = gamma
        self.beta = beta

    def _parse_llm_output(self, raw_output: str) -> InferenceResult:
        """解析 LLM JSON 输出"""
        try:
            # 提取 JSON 块
            json_match = re.search(r'\{[^{}]*\}', raw_output, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                level = data.get("level", "L2")
                if level not in self.LEVEL_INDEX:
                    level = "L2"
                return InferenceResult(
                    level=level,
                    confidence=float(data.get("confidence", 0.5)),
                    reason=data.get("reason", ""),
                    category=data.get("category", ""),
                )
        except (json.JSONDecodeError, ValueError):
            pass
        return InferenceResult(level="L2", confidence=0.3, reason="解析失败")

    def mc_dropout_inference(
        self, llm_fn, prompt: str
    ) -> List[InferenceResult]:
        """阶段1: MC-Dropout 多轮推理

        保持 Dropout 激活，进行 T 轮独立前向传播。
        4-bit 量化权重固定，仅 Dropout 掩码变化。

        Args:
            llm_fn: LLM 推理函数 (prompt → raw_text)
            prompt: 结构化提示文本

        Returns:
            T 轮推理结果列表
        """
        results = []
        for t in range(self.T):
            # 每轮使用不同的随机 Dropout 掩码
            raw_output = llm_fn(prompt, seed=t)
            result = self._parse_llm_output(raw_output)
            results.append(result)
        return results

    def compute_prediction_distribution(
        self, results: List[InferenceResult]
    ) -> np.ndarray:
        """计算预测分布均值

        P̂(ℓ|d) = (1/T) Σ_{t=1}^{T} P_t(ℓ|d, ε_t)

        Returns:
            4维概率分布 [P(L1), P(L2), P(L3), P(L4)]
        """
        counts = np.zeros(4)
        for r in results:
            idx = self.LEVEL_INDEX.get(r.level, 1)
            counts[idx] += 1
        return counts / (len(results) + 1e-8)

    def compute_prediction_variance(
        self, results: List[InferenceResult]
    ) -> np.ndarray:
        """计算预测方差

        Var[P̂(ℓ|d)] = (1/T) Σ P_t²  - P̂²

        Returns:
            4维方差向量
        """
        T = len(results)
        mean_dist = self.compute_prediction_distribution(results)

        # 每轮的 one-hot 分布
        per_round = np.zeros((T, 4))
        for t, r in enumerate(results):
            idx = self.LEVEL_INDEX.get(r.level, 1)
            per_round[t, idx] = 1.0

        mean_sq = np.mean(per_round ** 2, axis=0)
        variance = mean_sq - mean_dist ** 2
        return variance

    def compute_uncertainty(self, results: List[InferenceResult]) -> float:
        """阶段2: 不确定性量化

        U(d) = H[P̂(ℓ|d)] + β · tr(Var[P̂(ℓ|d)])

        Returns:
            不确定性评分
        """
        dist = self.compute_prediction_distribution(results)
        variance = self.compute_prediction_variance(results)

        # 预测熵 H[P̂]
        nonzero = dist[dist > 0]
        entropy = -np.sum(nonzero * np.log(nonzero + 1e-12))

        # 方差迹
        var_trace = np.sum(variance)

        return entropy + self.beta * var_trace

    def majority_vote(self, results: List[InferenceResult]) -> Tuple[str, float, float]:
        """阶段3: 自洽性投票

        ℓ* = argmax_ℓ Σ I(ℓ_t = ℓ)   —— 多数投票
        ρ = (1/T) Σ I(ℓ_t = ℓ*)       —— 一致性比率
        c̄ = Σ_{ℓ_t=ℓ*} c_t / Σ I(ℓ_t = ℓ*)  —— 置信度加权

        Returns:
            (最终分级, 一致性比率, 置信度加权评分)
        """
        # 多数投票
        level_counts = Counter(r.level for r in results)
        majority_level = level_counts.most_common(1)[0][0]

        # 一致性比率
        T = len(results)
        rho = level_counts[majority_level] / T

        # 置信度加权评分
        matching_confidences = [r.confidence for r in results if r.level == majority_level]
        c_bar = np.mean(matching_confidences) if matching_confidences else 0.0

        return majority_level, float(rho), float(c_bar)

    def verify(
        self, llm_fn, prompt: str
    ) -> GradingDecision:
        """完整的自洽性校验流程（算法 alg:context-grading 阶段3-5）

        Args:
            llm_fn: LLM 推理函数
            prompt: 结构化提示

        Returns:
            分级决策结果
        """
        # 阶段1: MC-Dropout 多轮推理
        results = self.mc_dropout_inference(llm_fn, prompt)
        all_predictions = [r.level for r in results]

        # 阶段2: 不确定性量化
        uncertainty = self.compute_uncertainty(results)
        if uncertainty > self.gamma:
            return GradingDecision(
                level="UNKNOWN",
                confidence=0.0,
                consistency_ratio=0.0,
                uncertainty=uncertainty,
                is_accepted=False,
                needs_human_review=True,
                reason=f"高不确定性 U={uncertainty:.3f} > γ={self.gamma}",
                all_predictions=all_predictions,
            )

        # 阶段3: 自洽性投票
        level, rho, c_bar = self.majority_vote(results)

        # 双阈值决策
        is_accepted = (rho >= self.rho_min) and (c_bar >= self.c_min)
        needs_review = not is_accepted

        # 获取理由（取多数票中置信度最高的结果的理由）
        matching = [r for r in results if r.level == level]
        best_reason = max(matching, key=lambda r: r.confidence).reason if matching else ""

        return GradingDecision(
            level=level,
            confidence=c_bar,
            consistency_ratio=rho,
            uncertainty=uncertainty,
            is_accepted=is_accepted,
            needs_human_review=needs_review,
            reason=best_reason,
            all_predictions=all_predictions,
        )

    @staticmethod
    def pac_bound(p: float, T: int) -> float:
        """PAC 理论保证

        P(correct) = Σ_{k=⌈T/2⌉}^{T} C(T,k) · p^k · (1-p)^{T-k}

        当单次准确率 p>0.5 时，多数投票放大正确率。
        例如 T=5, p=0.7 → 0.837；p=0.8 → 0.942

        Args:
            p: 单次推理准确率
            T: 投票轮次

        Returns:
            多数投票后的正确概率
        """
        from math import comb
        threshold = T // 2 + 1
        prob = sum(
            comb(T, k) * (p ** k) * ((1 - p) ** (T - k))
            for k in range(threshold, T + 1)
        )
        return prob
