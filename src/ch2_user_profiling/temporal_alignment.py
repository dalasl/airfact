"""
多源异构数据时空对齐

算法 alg:temporal-align 实现。
将文件操作事件与文档语义解析结果在 δ=5秒 时间窗口内关联，
实现行为日志与内容语义的时空对齐。

论文对应：
    - 算法 alg:temporal-align
    - 事件六元组结构 s_i = (type, timestamp, subject, object, action, context)
"""

import hashlib
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np


class EventType(Enum):
    """事件类型枚举"""
    FILE_ACCESS = "FILE_ACCESS"
    PROCESS_CREATE = "PROCESS_CREATE"
    NET_CONNECT = "NET_CONNECT"
    CLIPBOARD = "CLIPBOARD"


class ActionType(Enum):
    """操作类型枚举"""
    READ = "READ"
    WRITE = "WRITE"
    DELETE = "DELETE"
    COPY = "COPY"
    SEND = "SEND"
    UPLOAD = "UPLOAD"
    DOWNLOAD = "DOWNLOAD"
    PRINT = "PRINT"


@dataclass
class RawEvent:
    """原始行为事件"""
    event_type: EventType
    timestamp: float              # 毫秒精度时间戳
    process: str                  # 进程名称
    process_path: str = ""        # 进程路径
    parent_process: str = ""      # 父进程
    target_path: str = ""         # 目标文件路径
    src_addr: str = ""            # 源地址
    dst_addr: str = ""            # 目标地址
    protocol: str = ""            # 网络协议
    content_type: str = ""        # 内容类型
    data_length: int = 0          # 数据长度
    user_id: str = ""             # 用户 ID
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuredEvent:
    """结构化事件六元组

    s_i = (type, timestamp, subject, object, action, context)
    """
    event_type: EventType
    timestamp: float
    subject: Dict[str, str]       # {user_id, process_name, process_path, parent_process}
    obj: Dict[str, Any]           # {target_path, file_hash, file_size, access_permission}
    action: ActionType
    context: Dict[str, Any]       # {device_state, network_env, user_token, integrity_level}

    # 语义标注（来自文档解析缓存关联）
    semantic_vector: Optional[np.ndarray] = None  # 768维
    keywords: Optional[List[str]] = None


class TemporalAligner:
    """多源异构数据时空对齐器

    核心逻辑：
    1. 过滤系统进程（svchost, systemd 等）
    2. 结构化为六元组
    3. 文件访问事件与文档语义缓存关联（δ=5s 时间窗口）
    4. 异步解析结果延迟回填

    Args:
        doc_cache: 文档语义缓存引用
        system_whitelist: 系统进程白名单
        delta_seconds: 时间窗口大小（秒）
        parse_doc_fn: 异步文档解析回调函数
    """

    DEFAULT_SYSTEM_WHITELIST = {
        "svchost.exe", "systemd", "System", "csrss.exe",
        "lsass.exe", "services.exe", "wininit.exe",
        "smss.exe", "winlogon.exe", "spoolsv.exe",
    }

    def __init__(
        self,
        doc_cache=None,
        system_whitelist: Set[str] = None,
        delta_seconds: float = 5.0,
        parse_doc_fn: Callable = None,
    ):
        self.doc_cache = doc_cache
        self.system_whitelist = system_whitelist or self.DEFAULT_SYSTEM_WHITELIST
        self.delta = delta_seconds
        self.parse_doc_fn = parse_doc_fn

        # 待回填队列
        self.pending_queue: deque = deque()

    def _is_system_process(self, process_name: str) -> bool:
        """检查是否为系统进程"""
        return process_name.lower() in {p.lower() for p in self.system_whitelist}

    def _structurize(self, event: RawEvent) -> StructuredEvent:
        """将原始事件结构化为六元组"""
        # 推断操作类型
        action = self._infer_action(event)

        return StructuredEvent(
            event_type=event.event_type,
            timestamp=event.timestamp,
            subject={
                "user_id": event.user_id,
                "process_name": event.process,
                "process_path": event.process_path,
                "parent_process": event.parent_process,
            },
            obj={
                "target_path": event.target_path,
                "file_hash": "",
                "file_size": event.data_length,
                "access_permission": event.metadata.get("access_permission", ""),
            },
            action=action,
            context={
                "device_state": event.metadata.get("device_state", "normal"),
                "network_env": event.metadata.get("network_env", "internal"),
                "user_token": event.metadata.get("user_token", ""),
                "integrity_level": event.metadata.get("integrity_level", "medium"),
            },
        )

    @staticmethod
    def _infer_action(event: RawEvent) -> ActionType:
        """从原始事件推断操作类型"""
        action_map = {
            "read": ActionType.READ,
            "write": ActionType.WRITE,
            "delete": ActionType.DELETE,
            "copy": ActionType.COPY,
            "send": ActionType.SEND,
            "upload": ActionType.UPLOAD,
        }
        raw_action = event.metadata.get("action", "read").lower()
        return action_map.get(raw_action, ActionType.READ)

    def _try_semantic_correlation(self, structured: StructuredEvent, event: RawEvent):
        """尝试与文档语义缓存关联

        若文件哈希在缓存中且时间差 ≤ δ，则直接附加语义信息；
        否则提交异步解析请求，加入待回填队列。
        """
        if event.event_type != EventType.FILE_ACCESS:
            return

        if not event.target_path:
            return

        # 计算文件内容哈希（与 DocumentParser.compute_doc_hash 保持一致）
        try:
            with open(event.target_path, "rb") as fh:
                doc_id = hashlib.sha256(fh.read()).hexdigest()
        except (OSError, IOError):
            doc_id = hashlib.sha256(event.target_path.encode()).hexdigest()
        structured.obj["file_hash"] = doc_id

        # 查找缓存
        if self.doc_cache is not None:
            cached = self.doc_cache.get(doc_id)
            if cached is not None:
                time_diff = abs(event.timestamp - cached.timestamp)
                if time_diff <= self.delta:
                    # 缓存命中 + 时间窗口内 → 直接关联
                    structured.semantic_vector = cached.semantic_vector
                    structured.keywords = cached.keywords
                    return

        # 缓存未命中 → 提交异步解析，加入待回填队列
        if self.parse_doc_fn is not None:
            self.parse_doc_fn(event.target_path)

        structured.semantic_vector = None
        self.pending_queue.append((doc_id, structured))

    def _backfill_pending(self):
        """延迟回填：将已完成解析的文档语义附加到待处理事件"""
        if self.doc_cache is None:
            return

        remaining = deque()
        while self.pending_queue:
            doc_id, structured = self.pending_queue.popleft()
            cached = self.doc_cache.get(doc_id)
            if cached is not None:
                time_diff = abs(structured.timestamp - cached.timestamp)
                if time_diff <= self.delta:
                    structured.semantic_vector = cached.semantic_vector
                    structured.keywords = cached.keywords
                    continue
            remaining.append((doc_id, structured))
        self.pending_queue = remaining

    def align(self, events: List[RawEvent]) -> List[StructuredEvent]:
        """时空对齐主流程

        实现算法 alg:temporal-align 的完整逻辑。

        Args:
            events: 原始行为事件流 E = {e_1, e_2, ..., e_N}

        Returns:
            关联事件集合 A = {s_i}（含语义标注）
        """
        aligned_events = []

        for event in events:
            # Step 1: 过滤系统进程
            if self._is_system_process(event.process):
                continue

            # Step 2: 结构化为六元组
            structured = self._structurize(event)

            # Step 3: 语义关联
            self._try_semantic_correlation(structured, event)

            aligned_events.append(structured)

        # Step 4: 延迟回填
        self._backfill_pending()

        return aligned_events
