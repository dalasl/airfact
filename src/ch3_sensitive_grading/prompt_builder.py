"""
五层结构化提示构建器

按层级顺序构建 LLM 输入提示：
1. 系统角色层 — 定义任务边界
2. 静态画像层 — 注入用户画像先验
3. 动态 RAG 层 — 注入检索增强的修正案例
4. 数据输入层 — 提供文档核心信息
5. 输出约束层 — 强制 JSON 结构化输出

论文对应：第3章 3.2节 画像、经验与数据三维增强的结构化提示工程
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class UserProfile:
    """用户画像信息（来自第2章）"""
    user_id: str
    role: str                      # 角色标签（核心研发/外包人员/...）
    department: str                # 部门
    projects: List[str]            # 参与项目
    permissions: List[str]         # 权限范围
    business_keywords: List[str]   # 业务关键词
    cluster_id: int = -1           # 所属聚类


@dataclass
class CorrectionCase:
    """修正案例"""
    doc_summary: str               # 文档摘要
    original_level: str            # 系统原始分级 (L1-L4)
    corrected_level: str           # 管理员修正后分级
    correction_reason: str         # 修正理由
    attention_weight: float = 0.0  # Top-p 筛选后的归一化注意力权重


class PromptBuilder:
    """五层结构化提示构建器

    Args:
        system_role_text: 系统角色描述文本
        output_schema: 输出 JSON Schema 定义
    """

    DEFAULT_SYSTEM_ROLE = (
        "你是企业数据安全专家，任务是根据上下文判断文件敏感等级(L1-L4)。\n"
        "你必须综合考虑文件内容语义、操作用户的角色画像和历史修正案例，\n"
        "给出准确的敏感等级判定。\n\n"
        "等级定义：\n"
        "- L1（公开级）：公开信息，无访问限制\n"
        "- L2（内部级）：仅限组织内部使用\n"
        "- L3（机密级）：严格限制访问范围\n"
        "- L4（绝密级）：最高级别保护，仅限授权人员"
    )

    OUTPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "level": {"type": "string", "enum": ["L1", "L2", "L3", "L4"]},
            "category": {"type": "string", "description": "数据类别"},
            "reason": {"type": "string", "description": "判定理由"},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["level", "category", "reason", "confidence"],
    }

    def __init__(
        self,
        system_role_text: str = None,
        output_schema: Dict = None,
    ):
        self.system_role = system_role_text or self.DEFAULT_SYSTEM_ROLE
        self.output_schema = output_schema or self.OUTPUT_SCHEMA

    def _build_layer1_system_role(self) -> str:
        """第1层：系统角色层"""
        return f"System: {self.system_role}"

    def _build_layer2_static_profile(self, profile: UserProfile) -> str:
        """第2层：静态画像层 — 注入用户画像先验信息"""
        lines = [
            "Context: [用户画像]",
            f"  角色: {profile.role}",
            f"  部门: {profile.department}",
            f"  项目: {', '.join(profile.projects) if profile.projects else '无'}",
            f"  权限: {', '.join(profile.permissions) if profile.permissions else '基础权限'}",
            f"  高频业务关键词: {', '.join(profile.business_keywords[:10]) if profile.business_keywords else '无'}",
        ]
        return "\n".join(lines)

    def _build_layer3_dynamic_rag(self, cases: List[CorrectionCase]) -> str:
        """第3层：动态 RAG 层 — 注入检索增强的修正案例"""
        if not cases:
            return "Few-Shot: [无历史修正案例]"

        lines = ["Few-Shot: [参考修正案例]"]
        for i, case in enumerate(cases, 1):
            lines.append(f"  案例{i} (相关度权重: {case.attention_weight:.3f}):")
            lines.append(f"    文档: {case.doc_summary}")
            lines.append(f"    原始分级: {case.original_level} → 修正分级: {case.corrected_level}")
            lines.append(f"    修正理由: {case.correction_reason}")
        return "\n".join(lines)

    def _build_layer4_data_input(
        self, file_path: str, content_snippet: str, metadata: Dict[str, Any] = None
    ) -> str:
        """第4层：数据输入层 — 提供文档核心信息"""
        lines = [
            "Input: [待分级文档]",
            f"  文件路径: {file_path}",
            f"  内容片段: {content_snippet[:500]}",  # 截断保护
        ]
        if metadata:
            lines.append(f"  元数据: {json.dumps(metadata, ensure_ascii=False)}")
        return "\n".join(lines)

    def _build_layer5_output_constraint(self) -> str:
        """第5层：输出约束层 — 强制 JSON 结构化输出"""
        schema_str = json.dumps(self.output_schema, ensure_ascii=False, indent=2)
        return (
            "Output: 仅输出以下 JSON 格式，不要添加任何额外文字：\n"
            f"{schema_str}"
        )

    def build_prompt(
        self,
        profile: UserProfile,
        cases: List[CorrectionCase],
        file_path: str,
        content_snippet: str,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """构建完整的五层结构化提示

        Args:
            profile: 用户画像
            cases: Top-p 筛选后的修正案例（来自 RAG）
            file_path: 待分级文件路径
            content_snippet: 文件内容片段
            metadata: 附加元数据

        Returns:
            完整提示文本
        """
        layers = [
            self._build_layer1_system_role(),
            "",
            self._build_layer2_static_profile(profile),
            "",
            self._build_layer3_dynamic_rag(cases),
            "",
            self._build_layer4_data_input(file_path, content_snippet, metadata),
            "",
            self._build_layer5_output_constraint(),
        ]
        return "\n".join(layers)

    def build_zero_shot_prompt(
        self, file_path: str, content_snippet: str
    ) -> str:
        """构建零样本提示（消融实验：无画像、无 RAG）"""
        layers = [
            self._build_layer1_system_role(),
            "",
            self._build_layer4_data_input(file_path, content_snippet),
            "",
            self._build_layer5_output_constraint(),
        ]
        return "\n".join(layers)
