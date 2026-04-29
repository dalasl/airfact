"""
语法违例修正算法

算法 alg:grammar-correction 实现。
当约束解码遇到死端状态（可行 token 集为空）时，
回溯到最近可修复位置并重新采样。

论文对应：算法 alg:grammar-correction
"""

from typing import Callable, List, Optional, Set, Tuple

import numpy as np


class GrammarCorrector:
    """语法违例修正器

    Args:
        max_backtrack: 最大回溯深度
    """

    def __init__(self, max_backtrack: int = 50):
        self.max_backtrack = max_backtrack

    def correct(
        self,
        generated_tokens: List[str],
        state_history: List[int],
        dfa,
        llm_logits_fn: Callable = None,
        vocabulary: List[str] = None,
    ) -> Tuple[List[str], List[int]]:
        """语法违例修正

        1. 检查是否需要修正：V_t ≠ ∅ 则无需修正
        2. 回溯阶段：逐步回退找到最近可修复位置 t*
        3. 重采样阶段：在 t* 处选择替代 token 并继续生成

        Args:
            generated_tokens: 当前生成的 token 序列
            state_history: DFA 状态历史
            dfa: DFA 实例
            llm_logits_fn: LLM logits 获取函数
            vocabulary: 词表

        Returns:
            (修正后的 token 序列, 修正后的状态历史)
        """
        if not generated_tokens:
            return generated_tokens, state_history

        current_state = state_history[-1]
        feasible = dfa.get_feasible_tokens(current_state)

        # Step 1: 无需修正
        if feasible:
            return generated_tokens, state_history

        # Step 2: 回溯阶段 — 寻找最近可修复位置
        repair_pos = None
        for tau in range(len(generated_tokens) - 1, -1, -1):
            if tau >= len(state_history):
                continue

            state_at_tau = state_history[tau]
            feasible_at_tau = dfa.get_feasible_tokens(state_at_tau)

            # 排除当前选择的 token
            if tau + 1 < len(generated_tokens):
                current_token = generated_tokens[tau + 1]
                alternative = feasible_at_tau - {current_token}
            else:
                alternative = feasible_at_tau

            if alternative:
                repair_pos = tau
                break

            # 超过最大回溯深度
            if len(generated_tokens) - tau > self.max_backtrack:
                break

        if repair_pos is None:
            # 无法修复 → 返回截断序列
            return generated_tokens[:1], state_history[:2]

        # Step 3: 重采样阶段
        state_at_repair = state_history[repair_pos]
        feasible_alternatives = dfa.get_feasible_tokens(state_at_repair)

        # 排除导致死端的 token
        if repair_pos + 1 < len(generated_tokens):
            bad_token = generated_tokens[repair_pos + 1]
            feasible_alternatives = feasible_alternatives - {bad_token}

        if not feasible_alternatives:
            return generated_tokens[:repair_pos + 1], state_history[:repair_pos + 2]

        # 选择替代 token（按 LLM 概率或默认选第一个）
        if llm_logits_fn and vocabulary:
            # 基于 LLM 概率选择
            new_token = self._select_by_probability(
                feasible_alternatives, llm_logits_fn, vocabulary,
                generated_tokens[:repair_pos + 1]
            )
        else:
            new_token = sorted(feasible_alternatives)[0]

        # 状态转移
        new_state = dfa.transition(state_at_repair, new_token)

        # 构建修正后的序列
        corrected_tokens = generated_tokens[:repair_pos + 1] + [new_token]
        corrected_states = state_history[:repair_pos + 1] + [new_state]

        return corrected_tokens, corrected_states

    @staticmethod
    def _select_by_probability(
        candidates: Set[str],
        llm_logits_fn: Callable,
        vocabulary: List[str],
        context: List[str],
    ) -> str:
        """基于 LLM 概率分布选择替代 token

        y_{t*+1} = argmax_{v ∈ V'} P_LM(v | y_{1:t*}, x)
        """
        try:
            logits = llm_logits_fn(context)
            best_token = None
            best_prob = -np.inf

            for token in candidates:
                if token in vocabulary:
                    idx = vocabulary.index(token)
                    if logits[idx] > best_prob:
                        best_prob = logits[idx]
                        best_token = token

            return best_token or sorted(candidates)[0]
        except Exception:
            return sorted(candidates)[0]
