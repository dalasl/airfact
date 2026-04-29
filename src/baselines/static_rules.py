"""
静态规则基线分级器

论文 5.4.1 表 tab:grading-compare 对比基线之一。
基于 38 个敏感关键词正则匹配 + 文件属性组合规则 + 通道管控规则。

此基线代表传统 DLP 系统的规则驱动方法：
- 优点：延迟低（~2ms/文档），可解释
- 缺点：泛化差（依赖手工规则），无法处理语义变体

对应论文表 5-4 "Static-Rules" 行。
"""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class RuleMatch:
    """规则匹配结果"""
    rule_id: str
    category: str
    matched_text: str
    severity: str  # L1-L4


@dataclass
class StaticGradingResult:
    """静态规则分级结果

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
    matched_rules: List[RuleMatch] = None


class StaticRuleGrader:
    """静态规则分级器

    三类规则层叠：
    1. 关键词规则：38 个敏感关键词/短语正则匹配
    2. 文件属性规则：文件大小 + 扩展名 + 路径模式
    3. 通道管控规则：USB/HTTP/IM 通道识别

    分级策略：取所有匹配规则中最高等级。
    """

    # 38 个敏感关键词，按等级分组
    KEYWORD_RULES: Dict[str, List[str]] = {
        "L4": [
            r"绝密", r"top\s*secret", r"核心机密",
            r"密钥", r"private\s*key", r"secret\s*key",
            r"信用卡号", r"credit\s*card", r"SSN",
            r"身份证号", r"passport\s*number",
        ],
        "L3": [
            r"机密", r"confidential", r"薪酬",
            r"salary", r"财务报[表告]", r"financial\s*report",
            r"合同金额", r"contract\s*amount",
            r"客户名单", r"client\s*list",
            r"源代码", r"source\s*code",
            r"技术架构", r"architecture",
        ],
        "L2": [
            r"内部", r"internal\s*only", r"仅限内部",
            r"会议纪要", r"meeting\s*minutes",
            r"工作计划", r"work\s*plan",
            r"项目进度", r"project\s*status",
            r"通讯录", r"directory",
        ],
        "L1": [
            r"公开", r"public", r"新闻稿",
            r"press\s*release", r"产品手册",
        ],
    }

    # 文件属性规则
    HIGH_RISK_EXTENSIONS = {".xlsx", ".xls", ".docx", ".doc", ".pdf", ".pptx", ".key"}
    ARCHIVE_EXTENSIONS = {".zip", ".rar", ".7z", ".tar", ".gz"}
    CODE_EXTENSIONS = {".py", ".java", ".c", ".cpp", ".go", ".rs", ".ts", ".js"}

    # 高敏感路径模式
    SENSITIVE_PATH_PATTERNS = [
        (r"(?i)(finance|财务|accounting)", "L3"),
        (r"(?i)(hr|人力|人事|salary|薪酬)", "L3"),
        (r"(?i)(legal|法务|contract|合同)", "L3"),
        (r"(?i)(secret|classified|绝密|机密)", "L4"),
        (r"(?i)(source|src|code|代码)", "L2"),
        (r"(?i)(public|公开|release)", "L1"),
    ]

    # 大文件阈值（超过此值提升一级）
    LARGE_FILE_THRESHOLD_MB = 50

    def __init__(self):
        self._compiled_rules: Dict[str, List[re.Pattern]] = {}
        self._compile_rules()

    def _compile_rules(self):
        for level, patterns in self.KEYWORD_RULES.items():
            self._compiled_rules[level] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def grade(
        self,
        content: str,
        file_path: str = "",
        file_size_bytes: int = 0,
        channel: str = "",
    ) -> StaticGradingResult:
        """分级单个文档

        Args:
            content: 文档文本内容
            file_path: 文件路径
            file_size_bytes: 文件大小（字节）
            channel: 传输通道（usb/http/email/im）

        Returns:
            分级结果
        """
        matches: List[RuleMatch] = []

        # --- 阶段 1: 关键词匹配 ---
        for level, patterns in self._compiled_rules.items():
            for pattern in patterns:
                m = pattern.search(content)
                if m:
                    matches.append(RuleMatch(
                        rule_id=f"kw_{level}_{pattern.pattern[:20]}",
                        category="keyword",
                        matched_text=m.group()[:50],
                        severity=level,
                    ))

        # --- 阶段 2: 文件属性规则 ---
        if file_path:
            ext = os.path.splitext(file_path)[1].lower()

            # 代码文件默认 L2
            if ext in self.CODE_EXTENSIONS:
                matches.append(RuleMatch(
                    rule_id="attr_code_ext",
                    category="file_attribute",
                    matched_text=f"code extension: {ext}",
                    severity="L2",
                ))

            # 压缩包提升一级（可能打包敏感内容）
            if ext in self.ARCHIVE_EXTENSIONS and file_size_bytes > 10 * 1024 * 1024:
                matches.append(RuleMatch(
                    rule_id="attr_large_archive",
                    category="file_attribute",
                    matched_text=f"large archive: {file_size_bytes // (1024*1024)}MB",
                    severity="L3",
                ))

            # 路径模式
            for pattern, level in self.SENSITIVE_PATH_PATTERNS:
                if re.search(pattern, file_path):
                    matches.append(RuleMatch(
                        rule_id=f"path_{level}",
                        category="path_pattern",
                        matched_text=file_path[:80],
                        severity=level,
                    ))

        # 大文件提升
        if file_size_bytes > self.LARGE_FILE_THRESHOLD_MB * 1024 * 1024:
            matches.append(RuleMatch(
                rule_id="attr_large_file",
                category="file_attribute",
                matched_text=f"large file: {file_size_bytes // (1024*1024)}MB",
                severity="L2",
            ))

        # --- 阶段 3: 通道管控规则 ---
        if channel:
            channel_risk = self._assess_channel_risk(channel)
            if channel_risk:
                matches.append(channel_risk)

        # --- 决策: 取最高等级 ---
        level_order = {"L4": 4, "L3": 3, "L2": 2, "L1": 1}
        if not matches:
            final_level = "L1"
            reason = "无匹配规则，默认公开"
        else:
            final_level = max(matches, key=lambda m: level_order.get(m.severity, 0)).severity
            top_matches = [m for m in matches if m.severity == final_level]
            reason = f"匹配 {len(matches)} 条规则，最高级别 {final_level}: " + \
                     "; ".join(m.matched_text[:30] for m in top_matches[:3])

        confidence = min(0.5 + 0.1 * len(matches), 0.95)

        return StaticGradingResult(
            level=final_level,
            confidence=confidence,
            reason=reason,
            all_predictions=[final_level],
            matched_rules=matches,
        )

    def _assess_channel_risk(self, channel: str) -> Optional[RuleMatch]:
        """通道风险评估"""
        channel = channel.lower()
        if channel in ("usb", "removable"):
            return RuleMatch(
                rule_id="channel_usb",
                category="channel",
                matched_text="USB/可移动存储",
                severity="L3",
            )
        elif channel in ("http", "cloud", "webmail"):
            return RuleMatch(
                rule_id="channel_http",
                category="channel",
                matched_text="HTTP/云端上传",
                severity="L2",
            )
        elif channel in ("im", "wechat", "slack"):
            return RuleMatch(
                rule_id="channel_im",
                category="channel",
                matched_text="即时通讯",
                severity="L2",
            )
        return None

    def grade_batch(
        self,
        documents: List[Dict],
    ) -> List[StaticGradingResult]:
        """批量分级

        Args:
            documents: [{content, file_path, file_size_bytes, channel}, ...]

        Returns:
            分级结果列表
        """
        return [
            self.grade(
                content=doc.get("content", ""),
                file_path=doc.get("file_path", ""),
                file_size_bytes=doc.get("file_size_bytes", 0),
                channel=doc.get("channel", ""),
            )
            for doc in documents
        ]
