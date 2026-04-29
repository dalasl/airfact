# src/ch4_rule_generation/__init__.py
"""
第4章：语义驱动的检测规则生成与场景自适应执行方法

模块功能：
- 四元组中间表示 IR = <u, o, a, c>
- 语法约束解码器（DFA + Grammar-Aware FSM）
- 语法违例修正算法
- 模板合成引擎（Z3 形式化验证）
- VQL 编译器
- 场景自适应决策（画像基线 + 三维偏离 + 动态白名单）
"""

from .quadruple_ir import QuadrupleIR
from .constrained_decoder import ConstrainedDecoder
from .grammar_correction import GrammarCorrector
from .template_synthesizer import TemplateSynthesizer
from .vql_compiler import VQLCompiler
from .adaptive_decision import AdaptiveDecisionEngine

__all__ = [
    "QuadrupleIR",
    "ConstrainedDecoder",
    "GrammarCorrector",
    "TemplateSynthesizer",
    "VQLCompiler",
    "AdaptiveDecisionEngine",
]
