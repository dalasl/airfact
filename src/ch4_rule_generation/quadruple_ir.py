"""
四元组中间表示

将自然语言安全策略解析为结构化的四元组 IR：
    i = <u, o, a, c>
    - u: 主体约束（角色、部门、项目、密级）
    - o: 客体约束（类型、路径、敏感度）
    - a: 动作约束（读/写/拷贝/上传等）
    - c: 上下文约束（时间、位置、设备、网络）

EBNF 文法定义及类型系统。

论文对应：第4章 4.3节
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ConstraintOperator(Enum):
    """约束比较运算符"""
    EQ = "=="
    NEQ = "!="
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    MATCHES = "matches"


@dataclass
class AttributeConstraint:
    """属性约束表达式"""
    attribute: str
    operator: ConstraintOperator
    value: Any

    def to_dict(self) -> Dict:
        return {
            "attribute": self.attribute,
            "operator": self.operator.value,
            "value": self.value,
        }


@dataclass
class SubjectConstraint:
    """主体约束 u"""
    constraints: List[AttributeConstraint] = field(default_factory=list)
    logic: str = "AND"  # AND / OR

    # 典型属性
    VALID_ATTRIBUTES = ["role", "department", "project", "clearance",
                         "user_id", "group", "rank"]


@dataclass
class ObjectConstraint:
    """客体约束 o"""
    constraints: List[AttributeConstraint] = field(default_factory=list)
    logic: str = "AND"

    VALID_ATTRIBUTES = ["type", "path", "sensitivity", "extension",
                         "size", "hash", "owner", "classification"]


@dataclass
class ActionConstraint:
    """动作约束 a"""
    constraints: List[AttributeConstraint] = field(default_factory=list)
    logic: str = "AND"

    VALID_ATTRIBUTES = ["read", "write", "copy", "upload", "download",
                         "print", "send", "delete", "rename", "move"]


@dataclass
class ContextConstraint:
    """上下文约束 c"""
    constraints: List[AttributeConstraint] = field(default_factory=list)
    logic: str = "AND"

    VALID_ATTRIBUTES = ["time", "location", "device", "network",
                         "ip_range", "work_hours", "vpn_status"]


@dataclass
class QuadrupleIR:
    """四元组中间表示

    i = <u, o, a, c>

    设计原则：
    1. 完备性 — 四维笛卡尔积覆盖所有合法策略空间
    2. 无歧义性 — 每个四元组唯一映射到一个语义解释
    3. 可组合性 — 支持逻辑运算符嵌套
    """
    subject: SubjectConstraint = field(default_factory=SubjectConstraint)
    obj: ObjectConstraint = field(default_factory=ObjectConstraint)
    action: ActionConstraint = field(default_factory=ActionConstraint)
    context: ContextConstraint = field(default_factory=ContextConstraint)

    policy_text: str = ""  # 原始自然语言策略文本

    def to_dict(self) -> Dict:
        """序列化为字典"""
        return {
            "subject": {
                "constraints": [c.to_dict() for c in self.subject.constraints],
                "logic": self.subject.logic,
            },
            "object": {
                "constraints": [c.to_dict() for c in self.obj.constraints],
                "logic": self.obj.logic,
            },
            "action": {
                "constraints": [c.to_dict() for c in self.action.constraints],
                "logic": self.action.logic,
            },
            "context": {
                "constraints": [c.to_dict() for c in self.context.constraints],
                "logic": self.context.logic,
            },
            "policy_text": self.policy_text,
        }

    def type_check(self) -> List[str]:
        """静态类型检查

        验证属性-值的类型一致性（公式 4.4, 4.5）。

        Returns:
            类型错误列表（空列表表示通过）
        """
        errors = []

        # 检查主体属性
        for c in self.subject.constraints:
            if c.attribute not in SubjectConstraint.VALID_ATTRIBUTES:
                errors.append(f"主体约束包含无效属性: {c.attribute}")

        # 检查客体属性
        for c in self.obj.constraints:
            if c.attribute not in ObjectConstraint.VALID_ATTRIBUTES:
                errors.append(f"客体约束包含无效属性: {c.attribute}")

        # 检查动作属性
        for c in self.action.constraints:
            if c.attribute not in ActionConstraint.VALID_ATTRIBUTES:
                errors.append(f"动作约束包含无效属性: {c.attribute}")

        # 检查上下文属性
        for c in self.context.constraints:
            if c.attribute not in ContextConstraint.VALID_ATTRIBUTES:
                errors.append(f"上下文约束包含无效属性: {c.attribute}")

        return errors

    @classmethod
    def from_natural_language(cls, policy: str) -> "QuadrupleIR":
        """从自然语言策略文本构建四元组（简化版规则匹配）

        完整版使用 LLM + 约束解码器（见 constrained_decoder.py）。
        此方法为回退/测试用途。
        """
        ir = cls(policy_text=policy)

        # 简单关键词匹配
        role_pattern = re.compile(r'(非?)?(核心研发|外包|财务|管理|运维|普通)(人员|员工)?')
        match = role_pattern.search(policy)
        if match:
            op = ConstraintOperator.NEQ if match.group(1) == "非" else ConstraintOperator.EQ
            ir.subject.constraints.append(
                AttributeConstraint("role", op, match.group(2))
            )

        # 检测敏感度约束
        level_pattern = re.compile(r'(机密|绝密|内部|公开)(级)?(及以上)?')
        match = level_pattern.search(policy)
        if match:
            level_map = {"公开": "L1", "内部": "L2", "机密": "L3", "绝密": "L4"}
            level = level_map.get(match.group(1), "L2")
            op = ConstraintOperator.GTE if match.group(3) else ConstraintOperator.EQ
            ir.obj.constraints.append(
                AttributeConstraint("sensitivity", op, level)
            )

        # 检测动作约束
        action_keywords = {
            "拷贝": "copy", "复制": "copy", "上传": "upload",
            "外发": "send", "打印": "print", "下载": "download",
        }
        for zh, en in action_keywords.items():
            if zh in policy:
                ir.action.constraints.append(
                    AttributeConstraint(en, ConstraintOperator.EQ, True)
                )

        # 检测时间约束
        time_pattern = re.compile(r'非?工作时间')
        if time_pattern.search(policy):
            is_non_work = "非" in policy
            ir.context.constraints.append(
                AttributeConstraint("work_hours", ConstraintOperator.EQ, not is_non_work)
            )

        # 检测设备约束
        if "USB" in policy.upper():
            ir.context.constraints.append(
                AttributeConstraint("device", ConstraintOperator.EQ, "usb")
            )

        return ir
