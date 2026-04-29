"""
事件总线 (Event Bus)

跨模块事件驱动通信框架，串联三层架构的异步协作。

事件流：

    终端代理                    服务端
    ┌──────────┐      EVENT_FILE_ACCESS       ┌──────────────┐
    │FileWatch ├──────────────────────────────►│ 感知层(ch2)  │
    │ProcWatch │                               │   画像匹配   │
    └──────────┘                               └──────┬───────┘
                                                      │ EVENT_PROFILE_MATCHED
                                               ┌──────▼───────┐
                                               │ 认知层(ch3)  │
                                               │   敏感分级   │
                                               └──────┬───────┘
                                                      │ EVENT_GRADING_DONE
                                               ┌──────▼───────┐
                                               │ 执行层(ch4)  │
                                               │   规则执行   │
                                               └──────┬───────┘
                    EVENT_RULE_PUSHED                  │
    ┌──────────┐◄─────────────────────────────────────┘
    │VQLExec   │
    └──────────┘

论文对应：
    - 图 1-2 系统总体架构图中的层间数据流
    - 4.4 节 场景自适应执行的联动机制
    - 3.4 节 敏感资产目录的变更通知
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# =====================================================================
# 事件类型定义
# =====================================================================

class EventType(Enum):
    """系统事件类型

    按论文三层架构分组：
    - FILE_*:       终端侧文件系统事件
    - PROFILE_*:    感知层（画像）事件
    - GRADING_*:    认知层（分级）事件
    - RULE_*:       执行层（规则）事件
    - DECISION_*:   决策结果事件
    - SYSTEM_*:     系统级事件
    """

    # --- 终端侧 ---
    FILE_ACCESS = "file.access"          # 文件被访问
    FILE_MODIFY = "file.modify"          # 文件被修改
    FILE_TRANSFER = "file.transfer"      # 文件传输（USB/邮件/IM/HTTP）
    FILE_DELETE = "file.delete"          # 文件被删除

    # --- 感知层 (ch2) ---
    PROFILE_MATCHED = "profile.matched"          # 画像匹配完成
    PROFILE_DRIFT = "profile.drift"              # 画像漂移检测
    PROFILE_UPDATED = "profile.updated"          # 画像已更新
    CLUSTER_CHANGED = "cluster.changed"          # 聚类结构变化

    # --- 认知层 (ch3) ---
    GRADING_DONE = "grading.done"                # 分级完成
    GRADING_ESCALATED = "grading.escalated"      # 分级升格（置信度不足）
    GRADING_REVIEW = "grading.review"            # 需人工复核
    CATALOG_UPDATED = "catalog.updated"          # 资产目录更新
    RAG_FEEDBACK = "rag.feedback"                # RAG 人工反馈

    # --- 执行层 (ch4) ---
    RULE_GENERATED = "rule.generated"            # 规则已生成
    RULE_PUSHED = "rule.pushed"                  # 规则已下发到终端
    RULE_TRIGGERED = "rule.triggered"            # 规则已触发
    RULE_EXPIRED = "rule.expired"                # 规则已过期

    # --- 决策 ---
    DECISION_BLOCK = "decision.block"            # 操作阻断
    DECISION_ALERT = "decision.alert"            # 告警提示
    DECISION_PASS = "decision.pass"              # 静默放行

    # --- 系统 ---
    SYSTEM_INIT = "system.init"                  # 系统初始化
    SYSTEM_SHUTDOWN = "system.shutdown"           # 系统关闭
    SYSTEM_ERROR = "system.error"                # 系统错误
    WHITELIST_ADDED = "whitelist.added"           # 白名单条目添加
    WHITELIST_EXPIRED = "whitelist.expired"       # 白名单条目过期


@dataclass
class Event:
    """系统事件"""
    event_type: EventType
    source: str                  # 事件来源模块
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    event_id: str = ""           # 自动生成
    correlation_id: str = ""     # 用于关联同一文件事件的完整处理链

    def __post_init__(self):
        if not self.event_id:
            import uuid
            self.event_id = str(uuid.uuid4())[:8]


@dataclass
class Subscription:
    """事件订阅"""
    event_type: EventType
    callback: Callable
    subscriber_name: str
    filter_fn: Optional[Callable] = None   # 可选的事件过滤函数
    priority: int = 0                       # 优先级（数值越大越先执行）


# =====================================================================
# 事件总线
# =====================================================================

class EventBus:
    """集中式事件总线

    实现发布-订阅模式，连接三层架构的模块间异步通信。

    特性：
    - 同步和异步发布
    - 事件过滤
    - 优先级订阅
    - 事件审计日志
    - 死信队列（处理失败的事件）

    典型使用：
        bus = EventBus()

        # 订阅
        bus.subscribe(EventType.FILE_ACCESS, perception_layer.on_file_access, "ch2")
        bus.subscribe(EventType.PROFILE_MATCHED, cognition_layer.on_profile, "ch3")
        bus.subscribe(EventType.GRADING_DONE, execution_layer.on_grading, "ch4")

        # 发布
        bus.publish(Event(
            event_type=EventType.FILE_ACCESS,
            source="terminal_agent",
            data={"file_path": "...", "user_id": "..."},
        ))
    """

    def __init__(self, enable_audit: bool = True, max_history: int = 10000):
        self._subscriptions: Dict[EventType, List[Subscription]] = defaultdict(list)
        self._lock = threading.Lock()
        self._enable_audit = enable_audit
        self._max_history = max_history
        self._event_history: List[Event] = []
        self._dead_letters: List[Dict] = []
        self._stats: Dict[str, int] = defaultdict(int)

    # -----------------------------------------------------------------
    # 订阅
    # -----------------------------------------------------------------

    def subscribe(
        self,
        event_type: EventType,
        callback: Callable,
        subscriber_name: str = "",
        filter_fn: Callable = None,
        priority: int = 0,
    ):
        """订阅事件

        Args:
            event_type: 要订阅的事件类型
            callback: 回调函数，签名 (event: Event) -> None
            subscriber_name: 订阅者名称（用于审计）
            filter_fn: 可选的事件过滤函数 (event: Event) -> bool
            priority: 优先级（数值越大越先执行）
        """
        sub = Subscription(
            event_type=event_type,
            callback=callback,
            subscriber_name=subscriber_name,
            filter_fn=filter_fn,
            priority=priority,
        )

        with self._lock:
            self._subscriptions[event_type].append(sub)
            # 按优先级排序（高优先级在前）
            self._subscriptions[event_type].sort(key=lambda s: -s.priority)

        logger.debug(f"[EventBus] {subscriber_name} 订阅 {event_type.value} (优先级={priority})")

    def unsubscribe(self, event_type: EventType, subscriber_name: str):
        """取消订阅"""
        with self._lock:
            self._subscriptions[event_type] = [
                s for s in self._subscriptions[event_type]
                if s.subscriber_name != subscriber_name
            ]

    def subscribe_all(self, callback: Callable, subscriber_name: str = ""):
        """订阅所有事件类型（用于审计/日志）"""
        for et in EventType:
            self.subscribe(et, callback, subscriber_name)

    # -----------------------------------------------------------------
    # 发布
    # -----------------------------------------------------------------

    def publish(self, event: Event):
        """同步发布事件

        按优先级顺序依次调用所有匹配的订阅者回调。
        """
        self._stats[f"published.{event.event_type.value}"] += 1

        if self._enable_audit:
            self._record_event(event)

        with self._lock:
            subscribers = list(self._subscriptions.get(event.event_type, []))

        for sub in subscribers:
            # 过滤检查
            if sub.filter_fn and not sub.filter_fn(event):
                continue

            try:
                sub.callback(event)
                self._stats[f"delivered.{event.event_type.value}"] += 1
            except Exception as e:
                logger.error(
                    f"[EventBus] 订阅者 {sub.subscriber_name} 处理 "
                    f"{event.event_type.value} 失败: {e}"
                )
                self._dead_letters.append({
                    "event": event,
                    "subscriber": sub.subscriber_name,
                    "error": str(e),
                    "timestamp": time.time(),
                })
                self._stats["dead_letters"] += 1

    def publish_async(self, event: Event):
        """异步发布事件（在新线程中处理）"""
        thread = threading.Thread(target=self.publish, args=(event,), daemon=True)
        thread.start()

    # -----------------------------------------------------------------
    # 便捷发布方法
    # -----------------------------------------------------------------

    def emit_file_access(
        self,
        file_path: str,
        user_id: str,
        action: str,
        process_name: str,
        correlation_id: str = "",
        **extra,
    ) -> Event:
        """发布文件访问事件"""
        event = Event(
            event_type=EventType.FILE_ACCESS,
            source="terminal_agent",
            data={
                "file_path": file_path,
                "user_id": user_id,
                "action": action,
                "process_name": process_name,
                **extra,
            },
            correlation_id=correlation_id,
        )
        self.publish(event)
        return event

    def emit_grading_done(
        self,
        file_path: str,
        level: str,
        confidence: float,
        correlation_id: str = "",
        **extra,
    ) -> Event:
        """发布分级完成事件"""
        event = Event(
            event_type=EventType.GRADING_DONE,
            source="ch3_grading",
            data={
                "file_path": file_path,
                "level": level,
                "confidence": confidence,
                **extra,
            },
            correlation_id=correlation_id,
        )
        self.publish(event)
        return event

    def emit_decision(
        self,
        action: str,
        risk_score: float,
        file_path: str,
        user_id: str,
        correlation_id: str = "",
        **extra,
    ) -> Event:
        """发布决策事件"""
        type_map = {
            "block": EventType.DECISION_BLOCK,
            "alert": EventType.DECISION_ALERT,
            "silent_monitoring": EventType.DECISION_PASS,
        }
        event = Event(
            event_type=type_map.get(action, EventType.DECISION_PASS),
            source="ch4_decision",
            data={
                "action": action,
                "risk_score": risk_score,
                "file_path": file_path,
                "user_id": user_id,
                **extra,
            },
            correlation_id=correlation_id,
        )
        self.publish(event)
        return event

    def emit_rule_pushed(
        self,
        rule_id: str,
        target_terminal: str,
        vql_script: str,
        correlation_id: str = "",
    ) -> Event:
        """发布规则下发事件"""
        event = Event(
            event_type=EventType.RULE_PUSHED,
            source="ch4_rule_gen",
            data={
                "rule_id": rule_id,
                "target": target_terminal,
                "vql_script": vql_script,
            },
            correlation_id=correlation_id,
        )
        self.publish(event)
        return event

    # -----------------------------------------------------------------
    # 事件历史与审计
    # -----------------------------------------------------------------

    def _record_event(self, event: Event):
        """记录事件到审计日志"""
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]

    def get_history(
        self,
        event_type: EventType = None,
        source: str = None,
        correlation_id: str = None,
        limit: int = 100,
    ) -> List[Event]:
        """查询事件历史

        Args:
            event_type: 按事件类型过滤
            source: 按来源过滤
            correlation_id: 按关联 ID 过滤（追踪完整处理链）
            limit: 返回数量上限
        """
        results = self._event_history

        if event_type:
            results = [e for e in results if e.event_type == event_type]
        if source:
            results = [e for e in results if e.source == source]
        if correlation_id:
            results = [e for e in results if e.correlation_id == correlation_id]

        return results[-limit:]

    def get_processing_chain(self, correlation_id: str) -> List[Event]:
        """获取文件事件的完整处理链

        通过 correlation_id 串联：
            FILE_ACCESS → PROFILE_MATCHED → GRADING_DONE → DECISION_*
        """
        return self.get_history(correlation_id=correlation_id, limit=100)

    def get_stats(self) -> Dict[str, Any]:
        """获取事件总线统计"""
        subscriber_count = sum(len(subs) for subs in self._subscriptions.values())
        return {
            "total_subscribers": subscriber_count,
            "subscriptions_by_type": {
                et.value: len(subs)
                for et, subs in self._subscriptions.items()
                if subs
            },
            "event_counts": dict(self._stats),
            "history_size": len(self._event_history),
            "dead_letters": len(self._dead_letters),
        }

    def get_dead_letters(self, limit: int = 50) -> List[Dict]:
        """获取死信（处理失败的事件）"""
        return self._dead_letters[-limit:]

    def clear_history(self):
        """清空事件历史"""
        self._event_history.clear()
        self._dead_letters.clear()
        self._stats.clear()


# =====================================================================
# 预置事件处理器
# =====================================================================

class EventLogger:
    """事件日志记录器

    订阅所有事件并记录到日志文件，用于审计和事后分析。
    """

    def __init__(self, log_file: str = None):
        self.log_file = log_file
        self._log_entries: List[Dict] = []

    def handle(self, event: Event):
        """处理事件"""
        entry = {
            "event_id": event.event_id,
            "type": event.event_type.value,
            "source": event.source,
            "timestamp": event.timestamp,
            "correlation_id": event.correlation_id,
            "data_keys": list(event.data.keys()),
        }
        self._log_entries.append(entry)

        # 写入文件
        if self.log_file:
            import json
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except OSError:
                pass

    def get_entries(self, limit: int = 100) -> List[Dict]:
        return self._log_entries[-limit:]


class AlertAggregator:
    """告警聚合器

    将短时间内同一用户或同一文件的多次告警聚合为一条，避免告警风暴。
    """

    def __init__(self, window_seconds: float = 60.0, max_alerts_per_window: int = 5):
        self.window = window_seconds
        self.max_alerts = max_alerts_per_window
        self._recent_alerts: Dict[str, List[float]] = defaultdict(list)
        self._suppressed_count: int = 0

    def should_alert(self, key: str) -> bool:
        """检查是否应发送告警（滑动窗口去重）"""
        now = time.time()
        cutoff = now - self.window

        # 清理过期记录
        self._recent_alerts[key] = [
            t for t in self._recent_alerts[key] if t > cutoff
        ]

        if len(self._recent_alerts[key]) >= self.max_alerts:
            self._suppressed_count += 1
            return False

        self._recent_alerts[key].append(now)
        return True

    def handle_alert(self, event: Event):
        """处理告警事件"""
        user_id = event.data.get("user_id", "unknown")
        file_path = event.data.get("file_path", "unknown")
        key = f"{user_id}:{file_path}"

        if self.should_alert(key):
            logger.warning(
                f"[Alert] {event.event_type.value}: "
                f"用户={user_id} 文件={file_path} "
                f"风险={event.data.get('risk_score', 'N/A')}"
            )
        else:
            logger.debug(f"[Alert] 告警被抑制: {key} (窗口内已达上限)")


# =====================================================================
# 工厂函数
# =====================================================================

def create_default_bus(log_file: str = None) -> EventBus:
    """创建带默认处理器的事件总线

    预注册：
    - EventLogger: 全事件审计日志
    - AlertAggregator: 告警/阻断事件聚合
    """
    bus = EventBus()

    # 全事件日志
    event_logger = EventLogger(log_file=log_file)
    bus.subscribe_all(event_logger.handle, "event_logger")

    # 告警聚合
    aggregator = AlertAggregator()
    bus.subscribe(EventType.DECISION_ALERT, aggregator.handle_alert, "alert_aggregator")
    bus.subscribe(EventType.DECISION_BLOCK, aggregator.handle_alert, "alert_aggregator")

    return bus
