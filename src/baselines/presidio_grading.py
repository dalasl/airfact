"""
Microsoft Presidio NER 基线分级器

论文 5.4.1 表 tab:grading-compare 对比基线之一。
使用 Presidio 命名实体识别器检测 PII/敏感实体，
并基于实体类型和数量映射敏感等级。

此基线代表实体驱动的敏感数据检测方法：
- 优点：开箱即用，支持多语言 PII 检测
- 缺点：仅识别结构化实体（姓名/卡号/地址），
        无法理解业务语义（如"Q1 预算执行率"）

对应论文表 5-4 "Presidio-NER" 行。
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from presidio_analyzer import AnalyzerEngine, RecognizerResult
    _HAS_PRESIDIO = True
except ImportError:
    _HAS_PRESIDIO = False


# 实体类型到敏感等级的映射
ENTITY_SEVERITY_MAP = {
    # L4 - 绝密
    "CRYPTO": "L4",
    "MEDICAL_LICENSE": "L4",
    "US_SSN": "L4",
    "US_PASSPORT": "L4",
    "UK_NHS": "L4",
    "SG_NRIC_FIN": "L4",
    "AU_TFN": "L4",
    "AU_MEDICARE": "L4",

    # L3 - 机密
    "CREDIT_CARD": "L3",
    "US_BANK_NUMBER": "L3",
    "IBAN_CODE": "L3",
    "US_ITIN": "L3",
    "US_DRIVER_LICENSE": "L3",
    "IP_ADDRESS": "L3",

    # L2 - 内部
    "PERSON": "L2",
    "EMAIL_ADDRESS": "L2",
    "PHONE_NUMBER": "L2",
    "LOCATION": "L2",
    "DATE_TIME": "L2",
    "NRP": "L2",
    "ORGANIZATION": "L2",

    # L1 - 公开
    "URL": "L1",
    "DOMAIN_NAME": "L1",
}

# 实体数量触发阈值（超过则提升一级）
ENTITY_COUNT_THRESHOLDS = {
    "L2": 5,   # >= 5 个 L2 实体 → L3
    "L3": 3,   # >= 3 个 L3 实体 → L4
}


@dataclass
class EntityMatch:
    """检测到的实体"""
    entity_type: str
    text: str
    score: float
    start: int
    end: int
    severity: str


@dataclass
class PresidioGradingResult:
    """Presidio 分级结果

    与 GradingDecision 对齐的接口。
    """
    level: str
    confidence: float
    consistency_ratio: float = 1.0
    uncertainty: float = 0.0
    is_accepted: bool = True
    needs_human_review: bool = False
    reason: str = ""
    all_predictions: List[str] = None
    entities: List[EntityMatch] = None
    entity_counts: Dict[str, int] = None


class PresidioGrader:
    """Presidio NER 分级器

    Args:
        languages: 支持语言列表
        score_threshold: 实体识别置信度阈值
    """

    def __init__(
        self,
        languages: List[str] = None,
        score_threshold: float = 0.5,
    ):
        self.languages = languages or ["en"]
        self.score_threshold = score_threshold
        self._analyzer = None

    def _init_analyzer(self):
        """延迟初始化 Presidio 分析器"""
        if self._analyzer is not None:
            return

        if not _HAS_PRESIDIO:
            return

        self._analyzer = AnalyzerEngine()

    def grade(self, content: str) -> PresidioGradingResult:
        """分级单个文档

        Args:
            content: 文档文本内容

        Returns:
            分级结果
        """
        if not _HAS_PRESIDIO:
            return self._regex_fallback_grade(content)

        self._init_analyzer()

        # 运行 Presidio 分析
        results: List[RecognizerResult] = self._analyzer.analyze(
            text=content,
            language=self.languages[0],
            score_threshold=self.score_threshold,
        )

        # 转换为 EntityMatch
        entities = []
        for r in results:
            severity = ENTITY_SEVERITY_MAP.get(r.entity_type, "L1")
            entities.append(EntityMatch(
                entity_type=r.entity_type,
                text=content[r.start:r.end][:30],
                score=r.score,
                start=r.start,
                end=r.end,
                severity=severity,
            ))

        return self._compute_level(entities)

    def _regex_fallback_grade(self, content: str) -> PresidioGradingResult:
        """Presidio 不可用时的正则回退

        使用简化正则检测常见 PII 模式。
        """
        entities = []

        patterns = [
            (r"\b\d{3}-\d{2}-\d{4}\b", "US_SSN", "L4"),
            (r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "CREDIT_CARD", "L3"),
            (r"\b[A-Z]{2}\d{2}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{2}\b", "IBAN_CODE", "L3"),
            (r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "IP_ADDRESS", "L3"),
            (r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", "EMAIL_ADDRESS", "L2"),
            (r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}\b", "PHONE_NUMBER", "L2"),
            (r"\b\d{17}[\dXx]\b", "CN_ID_CARD", "L4"),  # 中国身份证
            (r"\b\d{3}-\d{8}\b|\b\d{4}-\d{7}\b", "CN_PHONE", "L2"),  # 中国座机
            (r"(?i)\b(passport|护照)\s*(?:no|号)?[:\s]*[A-Z0-9]{6,9}\b", "PASSPORT", "L4"),
            (r"(?i)\b(salary|薪[酬资]|工资)\s*[:：]?\s*\d+", "SALARY_INFO", "L3"),
        ]

        for pattern, entity_type, severity in patterns:
            for m in re.finditer(pattern, content):
                entities.append(EntityMatch(
                    entity_type=entity_type,
                    text=m.group()[:30],
                    score=0.8,
                    start=m.start(),
                    end=m.end(),
                    severity=severity,
                ))

        return self._compute_level(entities)

    def _compute_level(self, entities: List[EntityMatch]) -> PresidioGradingResult:
        """从检测到的实体计算最终敏感等级"""
        level_order = {"L4": 4, "L3": 3, "L2": 2, "L1": 1}

        if not entities:
            return PresidioGradingResult(
                level="L1",
                confidence=0.5,
                reason="未检测到敏感实体",
                all_predictions=["L1"],
                entities=[],
                entity_counts={},
            )

        # 统计各等级实体数
        severity_counts = Counter(e.severity for e in entities)
        entity_type_counts = Counter(e.entity_type for e in entities)

        # 基础等级：取最高
        base_level = max(entities, key=lambda e: level_order.get(e.severity, 0)).severity

        # 数量提升规则
        final_level = base_level
        for level, threshold in ENTITY_COUNT_THRESHOLDS.items():
            if severity_counts.get(level, 0) >= threshold:
                level_num = level_order[level]
                upgrade_level = {2: "L3", 3: "L4"}.get(level_num)
                if upgrade_level and level_order.get(upgrade_level, 0) > level_order[final_level]:
                    final_level = upgrade_level

        avg_score = sum(e.score for e in entities) / len(entities)
        confidence = min(avg_score * (1 + 0.05 * len(entities)), 0.95)

        top_entities = sorted(entities, key=lambda e: level_order.get(e.severity, 0), reverse=True)[:3]
        reason = (
            f"检测到 {len(entities)} 个敏感实体 "
            f"({', '.join(f'{k}:{v}' for k, v in severity_counts.most_common(3))}): "
            + "; ".join(f"{e.entity_type}='{e.text}'" for e in top_entities)
        )

        return PresidioGradingResult(
            level=final_level,
            confidence=confidence,
            reason=reason,
            all_predictions=[final_level],
            entities=entities,
            entity_counts=dict(entity_type_counts),
        )

    def grade_batch(self, contents: List[str]) -> List[PresidioGradingResult]:
        """批量分级"""
        return [self.grade(c) for c in contents]
