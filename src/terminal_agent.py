"""
终端代理 (Terminal Agent)

模拟论文中部署于终端侧的轻量级代理，负责：
1. 文件系统事件采集（Minifilter/eBPF 模拟）
2. 进程行为监控（IM/浏览器/邮件客户端）
3. VQL 规则本地执行引擎
4. 事件上报与策略同步

部署拓扑（论文图 4-3）：
    ┌─────────────┐           ┌──────────────┐
    │  终端代理    │ ── gRPC ──│  服务端       │
    │ (本文件)     │           │ (pipeline.py) │
    │ ─ FileWatch │           │ ─ 画像匹配    │
    │ ─ ProcWatch │           │ ─ 敏感分级    │
    │ ─ VQLExec   │           │ ─ 规则生成    │
    └─────────────┘           └──────────────┘

论文对应：
    - 表 5-17 终端侧资源占用（CPU < 5%, Memory 35-45MB）
    - 4.4 节 场景自适应执行
    - 图 4-3 系统部署架构
"""

import hashlib
import logging
import os
import queue
import re
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =====================================================================
# 事件数据结构
# =====================================================================

class FileAction(Enum):
    """文件操作类型"""
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    RENAME = "rename"
    READ = "read"
    COPY = "copy"
    MOVE = "move"


class ChannelType(Enum):
    """泄露通道类型（论文 5.4.2 节四通道）"""
    EMAIL = "email"         # 邮件通道
    IM = "im"               # 即时通讯通道
    USB = "usb"             # USB 外设通道
    HTTP = "http"           # HTTP/云盘通道
    LOCAL = "local"         # 本地操作


@dataclass
class FileEvent:
    """文件系统事件"""
    file_path: str
    action: FileAction
    process_name: str
    process_pid: int = 0
    timestamp: float = field(default_factory=time.time)
    file_size: int = 0
    file_hash: str = ""
    user_id: str = ""
    device_type: str = "local"      # local / usb / bluetooth
    network_env: str = "internal"   # internal / external / vpn
    is_work_hours: bool = True
    channel: ChannelType = ChannelType.LOCAL
    metadata: Dict = field(default_factory=dict)

    @property
    def file_extension(self) -> str:
        return Path(self.file_path).suffix.lower()

    @property
    def is_sensitive_extension(self) -> bool:
        """检查是否为敏感文件格式"""
        sensitive_exts = {
            ".docx", ".doc", ".xlsx", ".xls", ".pdf", ".pptx",
            ".csv", ".txt", ".json", ".xml", ".pem", ".key",
            ".cer", ".pfx", ".p12", ".sql", ".db", ".bak",
        }
        return self.file_extension in sensitive_exts


@dataclass
class ProcessInfo:
    """进程信息"""
    pid: int
    name: str
    exe_path: str = ""
    cmdline: str = ""
    parent_pid: int = 0
    start_time: float = field(default_factory=time.time)
    network_connections: List[str] = field(default_factory=list)


# =====================================================================
# 文件系统监控器（Minifilter/eBPF 模拟）
# =====================================================================

class FileSystemWatcher:
    """文件系统事件监控器

    模拟 Windows Minifilter（或 Linux eBPF）驱动的文件事件采集。
    实际部署中替换为平台原生实现：
    - Windows: Minifilter IRP 回调
    - Linux:   eBPF kprobe on vfs_read/vfs_write/vfs_rename

    论文对应：表 5-17 行为日志采集 <1% CPU, 12MB
    """

    def __init__(
        self,
        watch_dirs: List[str] = None,
        poll_interval: float = 1.0,
        include_patterns: List[str] = None,
        exclude_patterns: List[str] = None,
    ):
        self.watch_dirs = watch_dirs or []
        self.poll_interval = poll_interval
        self.include_patterns = include_patterns or ["*"]
        self.exclude_patterns = exclude_patterns or [
            "*.tmp", "*.log", "~$*", "*.swp", "Thumbs.db", ".DS_Store",
        ]

        self._event_queue: queue.Queue = queue.Queue(maxsize=10000)
        self._file_state: Dict[str, Tuple[float, int]] = {}  # path → (mtime, size)
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """启动监控线程"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        logger.info(f"[FileWatch] 已启动，监控 {len(self.watch_dirs)} 个目录")

    def stop(self):
        """停止监控"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("[FileWatch] 已停止")

    def get_events(self, max_count: int = 100, timeout: float = 0.1) -> List[FileEvent]:
        """批量获取事件（非阻塞）"""
        events = []
        for _ in range(max_count):
            try:
                ev = self._event_queue.get_nowait()
                events.append(ev)
            except queue.Empty:
                break
        return events

    def _watch_loop(self):
        """轮询监控循环（生产环境应使用 inotify/ReadDirectoryChangesW）"""
        # 初次扫描建立基线
        self._scan_baseline()

        while self._running:
            for watch_dir in self.watch_dirs:
                if not os.path.isdir(watch_dir):
                    continue
                self._scan_directory(watch_dir)
            time.sleep(self.poll_interval)

    def _scan_baseline(self):
        """建立文件状态基线"""
        for watch_dir in self.watch_dirs:
            if not os.path.isdir(watch_dir):
                continue
            for root, _, files in os.walk(watch_dir):
                for fname in files:
                    if self._should_exclude(fname):
                        continue
                    fpath = os.path.join(root, fname)
                    try:
                        stat = os.stat(fpath)
                        self._file_state[fpath] = (stat.st_mtime, stat.st_size)
                    except OSError:
                        pass

    def _scan_directory(self, watch_dir: str):
        """单次目录扫描"""
        current_files: Set[str] = set()

        for root, _, files in os.walk(watch_dir):
            for fname in files:
                if self._should_exclude(fname):
                    continue
                fpath = os.path.join(root, fname)
                current_files.add(fpath)

                try:
                    stat = os.stat(fpath)
                    current_state = (stat.st_mtime, stat.st_size)
                except OSError:
                    continue

                prev_state = self._file_state.get(fpath)

                if prev_state is None:
                    # 新文件
                    self._emit_event(fpath, FileAction.CREATE, stat.st_size)
                elif current_state != prev_state:
                    # 文件已修改
                    self._emit_event(fpath, FileAction.MODIFY, stat.st_size)

                self._file_state[fpath] = current_state

        # 检查已删除文件
        for fpath in list(self._file_state.keys()):
            if fpath.startswith(watch_dir) and fpath not in current_files:
                self._emit_event(fpath, FileAction.DELETE, 0)
                del self._file_state[fpath]

    def _emit_event(self, file_path: str, action: FileAction, file_size: int):
        """发送文件事件"""
        event = FileEvent(
            file_path=file_path,
            action=action,
            process_name="file_monitor",
            file_size=file_size,
            is_work_hours=self._is_work_hours(),
        )
        try:
            self._event_queue.put_nowait(event)
        except queue.Full:
            logger.warning("[FileWatch] 事件队列已满，丢弃事件")

    def _should_exclude(self, filename: str) -> bool:
        """检查文件是否应排除"""
        import fnmatch
        for pat in self.exclude_patterns:
            if fnmatch.fnmatch(filename, pat):
                return True
        return False

    @staticmethod
    def _is_work_hours() -> bool:
        """判断当前是否为工作时间"""
        import datetime
        now = datetime.datetime.now()
        return now.weekday() < 5 and 9 <= now.hour < 18


# =====================================================================
# 进程行为监控器
# =====================================================================

class ProcessWatcher:
    """进程行为监控器

    监控敏感进程（IM/邮件/浏览器）的文件访问行为。
    通过进程名和行为模式推断泄露通道类型。

    论文对应：
        - 5.4.2 节 IM 通道检测：监控 IM 进程对本地已分级文件的读取行为
        - 不需要解密聊天流量，不需要 IM 聊天记录数据集
    """

    # 进程→通道类型映射
    PROCESS_CHANNEL_MAP = {
        # 邮件客户端
        "outlook.exe": ChannelType.EMAIL,
        "thunderbird.exe": ChannelType.EMAIL,
        "foxmail.exe": ChannelType.EMAIL,
        # 即时通讯
        "wechat.exe": ChannelType.IM,
        "dingtalk.exe": ChannelType.IM,
        "feishu.exe": ChannelType.IM,
        "lark.exe": ChannelType.IM,
        "teams.exe": ChannelType.IM,
        "qq.exe": ChannelType.IM,
        "slack.exe": ChannelType.IM,
        # 浏览器（云盘上传）
        "chrome.exe": ChannelType.HTTP,
        "msedge.exe": ChannelType.HTTP,
        "firefox.exe": ChannelType.HTTP,
        # 文件管理器（USB 拷贝等）
        "explorer.exe": ChannelType.LOCAL,
    }

    # 高风险进程（自定义或未知进程）
    SUSPICIOUS_PROCESSES = {
        "curl.exe", "wget.exe", "scp.exe", "ftp.exe",
        "nc.exe", "ncat.exe", "rclone.exe", "winscp.exe",
    }

    def __init__(self):
        self._monitored_pids: Dict[int, ProcessInfo] = {}
        self._process_file_access: Dict[int, List[str]] = {}  # pid → [file_paths]

    def identify_channel(self, process_name: str, device_type: str = "local") -> ChannelType:
        """根据进程名和设备类型推断泄露通道

        论文核心论据：终端 EDR + VQL 通过监控 IM 进程对本地已分级
        敏感文件的读取行为实现检测，不需要解密聊天流量。

        Args:
            process_name: 发起操作的进程名
            device_type: 设备类型

        Returns:
            推断的泄露通道类型
        """
        # USB 设备优先判定
        if device_type in ("usb", "removable"):
            return ChannelType.USB

        pname = process_name.lower()

        # 精确匹配
        if pname in self.PROCESS_CHANNEL_MAP:
            return self.PROCESS_CHANNEL_MAP[pname]

        # 模糊匹配
        for known_proc, channel in self.PROCESS_CHANNEL_MAP.items():
            if known_proc.replace(".exe", "") in pname:
                return channel

        # 未知进程 → 默认本地
        return ChannelType.LOCAL

    def is_suspicious_process(self, process_name: str) -> bool:
        """判断是否为可疑进程"""
        return process_name.lower() in self.SUSPICIOUS_PROCESSES

    def record_file_access(self, pid: int, file_path: str):
        """记录进程对文件的访问"""
        if pid not in self._process_file_access:
            self._process_file_access[pid] = []
        self._process_file_access[pid].append(file_path)

    def get_access_frequency(self, pid: int, window_seconds: float = 60.0) -> int:
        """获取进程在时间窗口内的文件访问频率"""
        accesses = self._process_file_access.get(pid, [])
        return len(accesses)


# =====================================================================
# VQL 本地执行引擎
# =====================================================================

class VQLExecutor:
    """VQL 规则本地执行引擎

    在终端侧执行 VQL（Velociraptor Query Language）规则。
    敏感数据不上传，仅上报匹配结果（数据最小化原则）。

    论文对应：
        - 表 5-17 VQL 执行: 85ms, 5% CPU, 35MB
        - 4.3.4 节 双引擎编译
    """

    def __init__(self):
        self._active_rules: Dict[str, str] = {}      # rule_id → vql_script
        self._rule_stats: Dict[str, Dict] = {}        # rule_id → {hits, misses, errors}

    def load_rule(self, rule_id: str, vql_script: str) -> bool:
        """加载 VQL 规则

        Args:
            rule_id: 规则标识符
            vql_script: VQL 脚本文本

        Returns:
            是否加载成功
        """
        if not self._validate_vql(vql_script):
            logger.error(f"[VQL] 规则 {rule_id} 语法校验失败")
            return False

        self._active_rules[rule_id] = vql_script
        self._rule_stats[rule_id] = {"hits": 0, "misses": 0, "errors": 0, "load_time": time.time()}
        logger.info(f"[VQL] 规则 {rule_id} 已加载 (共 {len(self._active_rules)} 条)")
        return True

    def unload_rule(self, rule_id: str):
        """卸载规则"""
        self._active_rules.pop(rule_id, None)
        self._rule_stats.pop(rule_id, None)

    def evaluate(self, event: FileEvent) -> List[Dict]:
        """对事件评估所有活跃规则

        Args:
            event: 文件系统事件

        Returns:
            触发的规则列表 [{"rule_id": ..., "action": ..., "detail": ...}]
        """
        triggered = []

        for rule_id, vql in self._active_rules.items():
            try:
                match = self._evaluate_single(vql, event)
                if match:
                    self._rule_stats[rule_id]["hits"] += 1
                    triggered.append({
                        "rule_id": rule_id,
                        "action": self._extract_response_action(vql),
                        "detail": f"VQL 规则 {rule_id} 匹配",
                        "timestamp": time.time(),
                    })
                else:
                    self._rule_stats[rule_id]["misses"] += 1
            except Exception as e:
                self._rule_stats[rule_id]["errors"] += 1
                logger.warning(f"[VQL] 规则 {rule_id} 执行异常: {e}")

        return triggered

    def _evaluate_single(self, vql: str, event: FileEvent) -> bool:
        """评估单条 VQL 规则是否匹配事件

        简化实现：解析 WHERE 子句中的条件并逐条匹配。
        生产环境中由 Velociraptor agent 原生执行。
        """
        # 提取 WHERE 子句
        where_match = re.search(r"WHERE\s+(.+)$", vql, re.IGNORECASE)
        if not where_match:
            return True  # 无条件 → 全匹配

        conditions = where_match.group(1)

        # 逐条件检查（简化的条件求值）
        event_attrs = {
            "username": event.user_id,
            "userrole": event.user_id,
            "fullpath": event.file_path,
            "filetype": event.file_extension,
            "networkenv": event.network_env,
            "processname": event.process_name,
        }

        # 检查字符串相等条件
        eq_pattern = re.findall(r"(\w+)\s*=\s*'([^']*)'", conditions)
        for attr, value in eq_pattern:
            attr_lower = attr.lower()
            if attr_lower in event_attrs:
                if event_attrs[attr_lower].lower() != value.lower():
                    return False

        # 检查正则匹配条件
        regex_pattern = re.findall(r"(\w+)\s*=~\s*'([^']*)'", conditions)
        for attr, pattern in regex_pattern:
            attr_lower = attr.lower()
            if attr_lower in event_attrs:
                if not re.search(pattern, event_attrs[attr_lower], re.IGNORECASE):
                    return False

        # 检查敏感度条件
        sens_match = re.search(r"Sensitivity\s*(>=?|<=?|=)\s*(\d+)", conditions)
        if sens_match:
            # 简化：事件无敏感度信息时跳过此条件
            pass

        return True

    @staticmethod
    def _validate_vql(vql: str) -> bool:
        """VQL 基本语法校验"""
        vql_upper = vql.upper().strip()
        if not vql_upper.startswith("SELECT"):
            return False
        if "FROM" not in vql_upper:
            return False
        if vql.count("(") != vql.count(")"):
            return False
        if vql.count("'") % 2 != 0:
            return False
        return True

    @staticmethod
    def _extract_response_action(vql: str) -> str:
        """从 VQL 上下文推断响应动作"""
        vql_lower = vql.lower()
        if "sensitivity >= 4" in vql_lower or "sensitivity = 4" in vql_lower:
            return "block"
        elif "sensitivity >= 3" in vql_lower:
            return "alert"
        return "alert"

    def get_stats(self) -> Dict[str, Dict]:
        """获取规则执行统计"""
        return dict(self._rule_stats)

    @property
    def active_rule_count(self) -> int:
        return len(self._active_rules)


# =====================================================================
# 终端代理（集成类）
# =====================================================================

class TerminalAgent:
    """终端代理主类

    集成文件监控、进程监控和 VQL 执行三个子模块。
    充当终端侧的检测探针，与服务端 DLPPipeline 协同工作。

    典型使用：
        agent = TerminalAgent(
            user_id="vm2_bob",
            watch_dirs=["C:/Users/bob/Documents"],
        )

        # 注册事件回调（连接到服务端）
        agent.set_event_callback(pipeline.process_file_event)

        # 启动
        agent.start()

        # ... 运行期间自动采集事件、匹配规则、上报风险 ...

        agent.stop()

    部署配置：
        - CPU 占用: < 5% (论文表 5-17)
        - 内存占用: 35-45 MB
        - 事件延迟: < 1ms (采集) + 85ms (VQL 执行)
    """

    def __init__(
        self,
        user_id: str,
        watch_dirs: List[str] = None,
        poll_interval: float = 1.0,
        max_file_size_mb: float = 50.0,
    ):
        self.user_id = user_id
        self.max_file_size = int(max_file_size_mb * 1024 * 1024)

        # 子模块
        self.file_watcher = FileSystemWatcher(
            watch_dirs=watch_dirs or [],
            poll_interval=poll_interval,
        )
        self.process_watcher = ProcessWatcher()
        self.vql_executor = VQLExecutor()

        # 回调
        self._event_callback: Optional[Callable] = None
        self._alert_callback: Optional[Callable] = None

        # 事件处理线程
        self._running = False
        self._process_thread: Optional[threading.Thread] = None

        # 统计
        self._stats = {
            "events_collected": 0,
            "events_processed": 0,
            "events_blocked": 0,
            "events_alerted": 0,
            "start_time": 0.0,
        }

    def set_event_callback(self, callback: Callable):
        """设置事件上报回调（连接到服务端 pipeline）

        callback 签名应与 DLPPipeline.process_file_event 兼容:
            callback(file_path, file_content, user_id, action, process_name,
                     device_type, network_env, is_work_hours)
        """
        self._event_callback = callback

    def set_alert_callback(self, callback: Callable):
        """设置告警回调（发送到安全运营平台）"""
        self._alert_callback = callback

    def start(self):
        """启动终端代理"""
        if self._running:
            return

        self._running = True
        self._stats["start_time"] = time.time()

        # 启动文件监控
        self.file_watcher.start()

        # 启动事件处理线程
        self._process_thread = threading.Thread(target=self._event_loop, daemon=True)
        self._process_thread.start()

        logger.info(f"[Agent] 终端代理已启动 (用户={self.user_id})")

    def stop(self):
        """停止终端代理"""
        self._running = False
        self.file_watcher.stop()
        if self._process_thread:
            self._process_thread.join(timeout=5)
        logger.info(f"[Agent] 终端代理已停止 (处理事件={self._stats['events_processed']})")

    def push_rule(self, rule_id: str, vql_script: str) -> bool:
        """接收服务端下发的 VQL 规则

        论文场景：认知层分级 → 执行层生成规则 → 下发到终端执行
        """
        return self.vql_executor.load_rule(rule_id, vql_script)

    def inject_event(self, event: FileEvent):
        """手动注入事件（用于测试和演示）"""
        self.file_watcher._event_queue.put(event)

    def _event_loop(self):
        """事件处理循环"""
        while self._running:
            events = self.file_watcher.get_events(max_count=50)

            for event in events:
                self._stats["events_collected"] += 1
                self._process_event(event)

            time.sleep(0.1)

    def _process_event(self, event: FileEvent):
        """处理单个文件事件"""
        # 1. 补充事件元数据
        event.user_id = self.user_id
        event.channel = self.process_watcher.identify_channel(
            event.process_name, event.device_type
        )

        # 2. 本地 VQL 规则匹配
        triggered = self.vql_executor.evaluate(event)
        if triggered:
            for t in triggered:
                if t["action"] == "block":
                    self._stats["events_blocked"] += 1
                    logger.warning(f"[Agent] 🚫 阻断: {event.file_path} (规则 {t['rule_id']})")
                elif t["action"] == "alert":
                    self._stats["events_alerted"] += 1

                if self._alert_callback:
                    self._alert_callback(event, t)

        # 3. 上报到服务端（读取文件内容）
        if self._event_callback and event.is_sensitive_extension:
            try:
                content = self._read_file_content(event.file_path)
                if content is not None:
                    self._event_callback(
                        file_path=event.file_path,
                        file_content=content,
                        user_id=event.user_id,
                        action=event.action.value,
                        process_name=event.process_name,
                        device_type=event.device_type,
                        network_env=event.network_env,
                        is_work_hours=event.is_work_hours,
                    )
            except Exception as e:
                logger.error(f"[Agent] 上报失败: {e}")

        self._stats["events_processed"] += 1

    def _read_file_content(self, file_path: str) -> Optional[bytes]:
        """读取文件内容（带大小限制）"""
        try:
            if not os.path.exists(file_path):
                return None
            size = os.path.getsize(file_path)
            if size > self.max_file_size:
                logger.info(f"[Agent] 文件过大，跳过: {file_path} ({size / 1024 / 1024:.1f}MB)")
                return None
            with open(file_path, "rb") as f:
                return f.read()
        except (PermissionError, OSError):
            return None

    def get_status(self) -> Dict:
        """获取代理运行状态"""
        uptime = time.time() - self._stats["start_time"] if self._stats["start_time"] else 0
        return {
            "user_id": self.user_id,
            "running": self._running,
            "uptime_seconds": round(uptime, 1),
            "active_rules": self.vql_executor.active_rule_count,
            "events_collected": self._stats["events_collected"],
            "events_processed": self._stats["events_processed"],
            "events_blocked": self._stats["events_blocked"],
            "events_alerted": self._stats["events_alerted"],
            "watch_dirs": self.file_watcher.watch_dirs,
        }

    def compute_file_hash(self, file_path: str) -> str:
        """计算文件 SHA-256 哈希（用于资产目录去重）"""
        try:
            sha = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha.update(chunk)
            return sha.hexdigest()
        except (PermissionError, OSError):
            return ""
