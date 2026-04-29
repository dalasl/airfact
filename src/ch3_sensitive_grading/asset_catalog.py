"""
敏感资产目录维护

管理终端上所有文档的敏感等级记录，支持三种更新触发模式：
1. 内容变更触发：文件写操作 → 内容哈希比对 → 重新分级
2. 画像变更触发：用户画像版本升级 → 关联资产按优先级重评
3. 周期巡检触发：24h 定时 → 超过 7 天未更新的条目重评

关键设计：联合哈希 = Hash(content ∥ profile_version)
同一文档在不同用户画像下有独立分级记录。

论文对应：第3章 3.4节 敏感资产目录的动态维护
"""

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple


class UpdateTrigger(Enum):
    """更新触发类型"""
    CONTENT_CHANGE = "content_change"     # 文件内容变更
    PROFILE_CHANGE = "profile_change"     # 用户画像版本变更
    PERIODIC_PATROL = "periodic_patrol"   # 周期巡检


@dataclass
class AssetEntry:
    """资产目录条目"""
    joint_hash: str                # 联合哈希 Hash(content ∥ profile_version)
    content_hash: str              # 文件内容哈希
    profile_version: int           # 画像版本号
    file_path: str                 # 文件路径
    user_id: str                   # 关联用户
    sensitivity_level: str         # 敏感等级 L1-L4
    confidence: float              # 分级置信度
    reasoning: str                 # 分级理由（JSON）
    grading_timestamp: float       # 分级时间戳
    last_updated: float = field(default_factory=time.time)


class SensitiveAssetCatalog:
    """敏感资产目录

    Args:
        patrol_interval_hours: 巡检周期（小时），默认 24
        expire_days: 条目过期天数，默认 7
        priority_order: 重评优先级顺序
    """

    def __init__(
        self,
        patrol_interval_hours: int = 24,
        expire_days: int = 7,
        priority_order: List[str] = None,
    ):
        self.patrol_interval = patrol_interval_hours * 3600
        self.expire_seconds = expire_days * 86400
        self.priority_order = priority_order or ["L4", "L3", "L2", "L1"]

        # 目录存储
        self._catalog: Dict[str, AssetEntry] = {}

        # 变更事件回调（通知第4章规则生成模块）
        self._change_listeners: List[Callable] = []

    @staticmethod
    def compute_joint_hash(content_hash: str, profile_version: int) -> str:
        """计算联合哈希

        key = Hash(content ∥ profile_version)
        确保同一文档在不同画像下有独立记录。
        """
        raw = f"{content_hash}||{profile_version}"
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def compute_content_hash(content: bytes) -> str:
        """计算文件内容哈希 (SHA-256)"""
        return hashlib.sha256(content).hexdigest()

    def add_change_listener(self, callback: Callable):
        """注册变更事件监听器（供第4章规则模块订阅）"""
        self._change_listeners.append(callback)

    def _notify_change(self, entry: AssetEntry, old_level: str, new_level: str):
        """通知等级变更事件"""
        event = {
            "file_path": entry.file_path,
            "user_id": entry.user_id,
            "old_level": old_level,
            "new_level": new_level,
            "timestamp": time.time(),
        }
        for listener in self._change_listeners:
            try:
                listener(event)
            except Exception:
                pass

    def get(self, content_hash: str, profile_version: int) -> Optional[AssetEntry]:
        """查询资产条目"""
        joint_hash = self.compute_joint_hash(content_hash, profile_version)
        return self._catalog.get(joint_hash)

    def upsert(self, entry: AssetEntry) -> Tuple[bool, Optional[str]]:
        """插入或更新资产条目

        Returns:
            (是否有等级变更, 旧等级)
        """
        old_entry = self._catalog.get(entry.joint_hash)
        old_level = old_entry.sensitivity_level if old_entry else None

        self._catalog[entry.joint_hash] = entry

        # 检测等级变更
        if old_level and old_level != entry.sensitivity_level:
            self._notify_change(entry, old_level, entry.sensitivity_level)
            return True, old_level

        return False, old_level

    def handle_content_change(
        self,
        file_path: str,
        new_content: bytes,
        user_id: str,
        profile_version: int,
        grading_fn: Callable,
    ) -> AssetEntry:
        """内容变更触发的重新分级

        1. Minifilter 捕获写完成事件
        2. 计算新内容哈希
        3. 与缓存哈希比对
        4. 不匹配 → 完整分级流水线
        5. 匹配 → 复用缓存结果

        Args:
            file_path: 文件路径
            new_content: 新文件内容
            user_id: 用户 ID
            profile_version: 当前画像版本
            grading_fn: 分级回调 (content → (level, confidence, reason))
        """
        new_hash = self.compute_content_hash(new_content)
        existing = self.get(new_hash, profile_version)

        if existing is not None:
            # 哈希匹配 → 零资源消耗复用
            return existing

        # 内容变更 → 重新分级
        level, confidence, reason = grading_fn(new_content)

        entry = AssetEntry(
            joint_hash=self.compute_joint_hash(new_hash, profile_version),
            content_hash=new_hash,
            profile_version=profile_version,
            file_path=file_path,
            user_id=user_id,
            sensitivity_level=level,
            confidence=confidence,
            reasoning=reason,
            grading_timestamp=time.time(),
        )
        self.upsert(entry)
        return entry

    def handle_profile_change(
        self,
        user_id: str,
        new_profile_version: int,
        grading_fn: Callable,
    ) -> List[AssetEntry]:
        """画像变更触发的批量重评

        按优先级顺序处理：L4 → L3 → L2 → L1

        Args:
            user_id: 用户 ID
            new_profile_version: 新画像版本号
            grading_fn: 分级回调

        Returns:
            更新后的条目列表
        """
        # 找出该用户所有旧版本条目
        affected = [
            entry for entry in self._catalog.values()
            if entry.user_id == user_id and entry.profile_version < new_profile_version
        ]

        # 按优先级排序
        priority_map = {level: i for i, level in enumerate(self.priority_order)}
        affected.sort(key=lambda e: priority_map.get(e.sensitivity_level, 99))

        updated = []
        for entry in affected:
            # 使用新画像重新分级
            level, confidence, reason = grading_fn(entry.content_hash)

            new_entry = AssetEntry(
                joint_hash=self.compute_joint_hash(entry.content_hash, new_profile_version),
                content_hash=entry.content_hash,
                profile_version=new_profile_version,
                file_path=entry.file_path,
                user_id=user_id,
                sensitivity_level=level,
                confidence=confidence,
                reasoning=reason,
                grading_timestamp=time.time(),
            )
            self.upsert(new_entry)
            updated.append(new_entry)

        return updated

    def periodic_patrol(self, grading_fn: Callable) -> List[AssetEntry]:
        """周期巡检触发

        检查所有超过 7 天未更新的条目，批量重评。

        Returns:
            更新后的条目列表
        """
        now = time.time()
        expired = [
            entry for entry in self._catalog.values()
            if (now - entry.last_updated) > self.expire_seconds
        ]

        updated = []
        for entry in expired:
            level, confidence, reason = grading_fn(entry.content_hash)
            entry.sensitivity_level = level
            entry.confidence = confidence
            entry.reasoning = reason
            entry.grading_timestamp = now
            entry.last_updated = now
            self.upsert(entry)
            updated.append(entry)

        return updated

    def get_asset_list_by_level(self) -> Dict[str, List[AssetEntry]]:
        """获取按等级分层的资产列表（供第4章规则生成使用）"""
        result = {"L1": [], "L2": [], "L3": [], "L4": []}
        for entry in self._catalog.values():
            if entry.sensitivity_level in result:
                result[entry.sensitivity_level].append(entry)
        return result

    def __len__(self) -> int:
        return len(self._catalog)
