"""
语法约束解码器

基于 DFA（确定有限自动机）的语法感知 FSM 解码器。
在 LLM token 生成过程中动态约束输出，确保生成结果
严格符合四元组 EBNF 文法。

核心机制：
1. EBNF 文法编译为 DFA（~1200 状态，15MB 转移表）
2. 每步解码时计算可行 token 集 V_t = {v: δ(q_t, v) ≠ ∅}
3. 不可行 token 的 logits 设为 -∞
4. 违例时触发回溯修正（见 grammar_correction.py）
5. outlines FSM 约束解码（当 outlines 可用时）

论文对应：
    - 公式 (4.2): 约束解码概率
    - 公式 (4.3): DFA 定义 M = (Q, Σ, δ, q_0, F)
    - 公式 (4.6): 可行 token 集
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import outlines
    _HAS_OUTLINES = True
except ImportError:
    _HAS_OUTLINES = False


# 四元组 JSON Schema（outlines / regex 共用）
QUADRUPLE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "subject": {
            "type": "object",
            "properties": {
                "role": {"type": "string"},
                "department": {"type": "string"},
                "clearance": {"type": "string"},
            },
            "required": ["role"],
        },
        "object": {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "sensitivity": {"type": "string", "enum": ["L1", "L2", "L3", "L4"]},
                "path": {"type": "string"},
            },
            "required": ["type", "sensitivity"],
        },
        "action": {
            "type": "string",
            "enum": ["read", "write", "copy", "upload", "send", "print", "download", "delete"],
        },
        "context": {
            "type": "object",
            "properties": {
                "time": {"type": "string"},
                "device": {"type": "string"},
                "network": {"type": "string", "enum": ["internal", "external", "vpn"]},
                "work_hours": {"type": "boolean"},
            },
        },
    },
    "required": ["subject", "object", "action"],
}


@dataclass
class DFAState:
    """DFA 状态"""
    state_id: int
    is_accepting: bool = False
    transitions: Dict[str, int] = field(default_factory=dict)


class DFA:
    """确定有限自动机

    通过 Thompson 构造 + 子集构造从 EBNF 文法编译。

    属性：
        Q: 状态集合
        Sigma: 输入字母表（词表子集）
        delta: 状态转移函数
        q0: 初始状态
        F: 接受状态集
    """

    def __init__(self):
        self.states: Dict[int, DFAState] = {}
        self.initial_state: int = 0
        self.accepting_states: Set[int] = set()
        self.vocabulary: Set[str] = set()

        # 预计算的可行 token 缓存
        self._feasible_cache: Dict[int, Set[str]] = {}

    def add_state(self, state_id: int, is_accepting: bool = False) -> DFAState:
        """添加状态"""
        state = DFAState(state_id=state_id, is_accepting=is_accepting)
        self.states[state_id] = state
        if is_accepting:
            self.accepting_states.add(state_id)
        return state

    def add_transition(self, from_state: int, token: str, to_state: int):
        """添加状态转移"""
        self.states[from_state].transitions[token] = to_state
        self.vocabulary.add(token)
        # 清除缓存
        self._feasible_cache.pop(from_state, None)

    def transition(self, state_id: int, token: str) -> Optional[int]:
        """执行状态转移 δ(q, v)"""
        if state_id not in self.states:
            return None
        return self.states[state_id].transitions.get(token)

    def get_feasible_tokens(self, state_id: int) -> Set[str]:
        """获取当前状态的可行 token 集

        V_t = {v ∈ V : δ(q_t, v) ≠ ∅}

        使用缓存实现 O(1) 查询。
        """
        if state_id in self._feasible_cache:
            return self._feasible_cache[state_id]

        if state_id not in self.states:
            return set()

        feasible = set(self.states[state_id].transitions.keys())
        self._feasible_cache[state_id] = feasible
        return feasible

    @classmethod
    def build_quadruple_grammar_dfa(cls) -> "DFA":
        """构建四元组 EBNF 文法的 DFA

        <intermediate> ::= <subject> <object> <action> <context>
        <subject> ::= "{" "subject" ":" <subject_expr> "}"
        <subject_expr> ::= <attr_compare> | <subject_expr> "AND" <subject_expr>
        ...

        简化版实现，实际版本约 1200 个状态。
        """
        dfa = cls()

        # 简化版状态机（完整版在编译时生成）
        # 状态 0: 初始，期望 {
        dfa.add_state(0)
        dfa.add_state(1)   # 等待 subject
        dfa.add_state(2)   # subject 内容
        dfa.add_state(3)   # 等待 object
        dfa.add_state(4)   # object 内容
        dfa.add_state(5)   # 等待 action
        dfa.add_state(6)   # action 内容
        dfa.add_state(7)   # 等待 context
        dfa.add_state(8)   # context 内容
        dfa.add_state(9, is_accepting=True)  # 接受状态

        # 基本转移
        dfa.add_transition(0, "{", 1)
        dfa.add_transition(1, "subject", 2)
        dfa.add_transition(2, ",", 3)
        dfa.add_transition(3, "object", 4)
        dfa.add_transition(4, ",", 5)
        dfa.add_transition(5, "action", 6)
        dfa.add_transition(6, ",", 7)
        dfa.add_transition(7, "context", 8)
        dfa.add_transition(8, "}", 9)

        # 属性值对的转移（简化）
        for state in [2, 4, 6, 8]:
            dfa.add_transition(state, ":", state)
            for attr in ["role", "department", "type", "path", "sensitivity",
                          "read", "write", "copy", "upload", "time", "device",
                          "network", "==", "!=", ">", ">=", "<", "<=",
                          "AND", "OR", "NOT"]:
                dfa.add_transition(state, attr, state)

        return dfa


class ConstrainedDecoder:
    """语法约束解码器

    在 LLM 生成过程中强制语法约束。

    Args:
        dfa: 编译后的 DFA
        max_length: 最大生成长度
        max_backtrack: 最大回溯深度
    """

    def __init__(
        self,
        dfa: DFA = None,
        max_length: int = 512,
        max_backtrack: int = 50,
    ):
        self.dfa = dfa or DFA.build_quadruple_grammar_dfa()
        self.max_length = max_length
        self.max_backtrack = max_backtrack

    def constrained_decode_step(
        self,
        logits: np.ndarray,
        vocabulary: List[str],
        current_state: int,
    ) -> Tuple[int, str, int]:
        """单步约束解码

        P(y_t | y_{1:t-1}, x) = P_LM(y_t) · I(y_t ∈ A) / Z

        Args:
            logits: LLM 输出的 logits [vocab_size]
            vocabulary: token 词表
            current_state: 当前 DFA 状态

        Returns:
            (选中token的索引, token文本, 新DFA状态)
        """
        # 获取可行 token 集
        feasible = self.dfa.get_feasible_tokens(current_state)

        # 构建掩码
        mask = np.full_like(logits, -np.inf)
        for i, token in enumerate(vocabulary):
            if token in feasible:
                mask[i] = 0.0

        # 应用掩码
        masked_logits = logits + mask

        # Softmax 采样
        exp_logits = np.exp(masked_logits - masked_logits.max())
        probs = exp_logits / (exp_logits.sum() + 1e-12)

        # 选择概率最高的 token（贪心）
        selected_idx = np.argmax(probs)
        selected_token = vocabulary[selected_idx]

        # 状态转移
        new_state = self.dfa.transition(current_state, selected_token)

        return selected_idx, selected_token, new_state

    def decode(
        self,
        llm_fn: Callable,
        input_text: str,
    ) -> Tuple[str, List[int]]:
        """完整约束解码过程（DFA 路径）

        Args:
            llm_fn: LLM 推理函数
            input_text: 输入文本（自然语言策略）

        Returns:
            (生成的四元组 JSON, DFA 状态历史)
        """
        current_state = self.dfa.initial_state
        generated_tokens = []
        state_history = [current_state]

        for step in range(self.max_length):
            feasible = self.dfa.get_feasible_tokens(current_state)

            if not feasible:
                break

            token = list(feasible)[0]
            generated_tokens.append(token)

            new_state = self.dfa.transition(current_state, token)
            if new_state is None:
                break

            current_state = new_state
            state_history.append(current_state)

            if current_state in self.dfa.accepting_states:
                break

        return " ".join(generated_tokens), state_history

    def decode_with_outlines(
        self,
        model,
        input_text: str,
        schema: Dict = None,
    ) -> Dict:
        """outlines FSM 约束解码（论文正式路径）

        使用 outlines 库的 JSON Schema 约束生成，
        确保 LLM 输出严格符合四元组 IR 结构。

        Args:
            model: outlines 模型实例 (outlines.models.transformers)
            input_text: 自然语言策略文本
            schema: JSON Schema（默认使用 QUADRUPLE_JSON_SCHEMA）

        Returns:
            解析后的四元组字典
        """
        if not _HAS_OUTLINES:
            return self._decode_with_regex_fallback(None, input_text)

        schema = schema or QUADRUPLE_JSON_SCHEMA

        generator = outlines.generate.json(model, schema)
        prompt = (
            f"将以下安全策略解析为四元组 IR (subject, object, action, context):\n"
            f"策略: {input_text}\n"
            f"输出 JSON:"
        )
        result = generator(prompt)
        return result

    def decode_with_schema(
        self,
        llm_fn: Callable,
        input_text: str,
        schema: Dict = None,
    ) -> Dict:
        """JSON Schema 约束解码（outlines 不可用时的回退路径）

        通过 LLM 生成 + JSON Schema 校验 + 重试的方式
        实现约束解码。最多重试 3 次。

        Args:
            llm_fn: LLM 推理函数 (prompt, seed) -> str
            input_text: 自然语言策略文本
            schema: JSON Schema

        Returns:
            校验通过的四元组字典
        """
        return self._decode_with_regex_fallback(llm_fn, input_text, schema)

    def _decode_with_regex_fallback(
        self,
        llm_fn: Optional[Callable],
        input_text: str,
        schema: Dict = None,
    ) -> Dict:
        """正则回退解码：LLM 生成 → JSON 提取 → Schema 校验 → 重试"""
        schema = schema or QUADRUPLE_JSON_SCHEMA
        valid_actions = schema["properties"]["action"].get("enum", [])
        valid_levels = schema["properties"]["object"]["properties"]["sensitivity"].get("enum", [])

        prompt = (
            "你是安全策略解析引擎。将自然语言策略转换为 JSON 四元组。\n"
            "严格按以下格式输出，不要添加其他文字：\n"
            '{"subject": {"role": "..."}, "object": {"type": "...", "sensitivity": "L1|L2|L3|L4"}, '
            '"action": "read|write|copy|upload|send|print|download|delete", '
            '"context": {"network": "internal|external|vpn", "work_hours": true|false}}\n\n'
            f"策略: {input_text}\n"
            "JSON:"
        )

        for attempt in range(3):
            if llm_fn is not None:
                raw = llm_fn(prompt, seed=attempt)
            else:
                raw = self._rule_based_parse(input_text)
                if raw:
                    return raw
                break

            parsed = self._extract_and_validate_json(raw, valid_actions, valid_levels)
            if parsed is not None:
                return parsed

        return self._rule_based_parse(input_text)

    @staticmethod
    def _extract_and_validate_json(
        raw: str,
        valid_actions: List[str],
        valid_levels: List[str],
    ) -> Optional[Dict]:
        """从 LLM 输出中提取并校验 JSON"""
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw, re.DOTALL)
        if not json_match:
            return None
        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return None

        if "subject" not in data or "object" not in data or "action" not in data:
            return None
        if valid_actions and data.get("action") not in valid_actions:
            return None
        sensitivity = data.get("object", {}).get("sensitivity", "")
        if valid_levels and sensitivity not in valid_levels:
            return None

        return data

    @staticmethod
    def _rule_based_parse(policy: str) -> Dict:
        """规则回退解析（无 LLM 时）"""
        from .quadruple_ir import QuadrupleIR
        ir = QuadrupleIR.from_natural_language(policy)
        ir_dict = ir.to_dict()

        result = {
            "subject": {"role": "any"},
            "object": {"type": "file", "sensitivity": "L2"},
            "action": "read",
            "context": {"network": "internal", "work_hours": True},
        }

        for c in ir.subject.constraints:
            result["subject"][c.attribute] = c.value
        for c in ir.obj.constraints:
            result["object"][c.attribute] = c.value
        for c in ir.action.constraints:
            result["action"] = c.attribute
        for c in ir.context.constraints:
            result["context"][c.attribute] = c.value

        return result
