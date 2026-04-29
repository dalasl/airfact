"""
CERT r4.2 行为模拟器

基于 CERT Insider Threat r4.2 数据集的统计分布，
使用马尔可夫状态机驱动生成 30 天 × 3 台 VM 的差异化行为事件流。

输出:
- 原始事件: data/behavior_logs/{vm_id}/events.jsonl (~85000条)
- 操作序列: data/sequences/{anomaly,normal}/*.jsonl (~930条)

核心设计:
1. 三台 VM 差异化角色配置 (行政/财务/核心业务)
2. 第 8 天 VM-1 角色迁移 (行政 → 核心业务)
3. 42 个数据窃取场景注入异常序列
4. 四种泄露通道: 邮件/HTTP/USB/IM
"""

import json
import os
import random
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# ---- 常量与配置 ----

SIMULATION_DAYS = 30
WORK_HOURS = (8, 18)  # 工作时间 8:00-18:00
EVENT_TYPES = ["file", "email", "http", "device", "im"]

# 泄露通道及其异常/正常序列配额
CHANNEL_QUOTAS = {
    "email":  {"anomaly": 100, "normal": 150},
    "http":   {"anomaly": 95,  "normal": 140},
    "usb":    {"anomaly": 95,  "normal": 140},
    "im":     {"anomaly": 90,  "normal": 120},
}

# 异常难度分布
DIFFICULTY_DISTRIBUTION = {"easy": 0.35, "medium": 0.40, "hard": 0.25}

# 角色迁移日
ROLE_MIGRATION_DAY = 8


@dataclass
class VMConfig:
    """虚拟机角色配置"""
    vm_id: str
    role: str
    primary_levels: List[str]       # 主要接触的敏感等级
    daily_event_range: Tuple[int, int]  # 日均事件数范围
    event_type_weights: Dict[str, float]  # 各事件类型权重
    file_ops_per_day: Tuple[int, int]  # 日均文件操作数
    email_ops_per_day: Tuple[int, int]
    http_ops_per_day: Tuple[int, int]
    device_ops_per_day: Tuple[int, int]
    im_ops_per_day: Tuple[int, int]
    work_hour_bias: float = 0.85     # 工作时段内操作占比


# 三台 VM 的角色配置（基于 CERT r4.2 同岗位统计）
VM_CONFIGS = {
    "vm1": VMConfig(
        vm_id="vm1",
        role="office_staff",
        primary_levels=["L1", "L2"],
        daily_event_range=(650, 1000),
        event_type_weights={"file": 0.35, "email": 0.25, "http": 0.20, "device": 0.05, "im": 0.15},
        file_ops_per_day=(30, 50),
        email_ops_per_day=(15, 30),
        http_ops_per_day=(10, 25),
        device_ops_per_day=(2, 5),
        im_ops_per_day=(10, 20),
    ),
    "vm2": VMConfig(
        vm_id="vm2",
        role="finance",
        primary_levels=["L2", "L3"],
        daily_event_range=(750, 1100),
        event_type_weights={"file": 0.40, "email": 0.25, "http": 0.15, "device": 0.05, "im": 0.15},
        file_ops_per_day=(40, 60),
        email_ops_per_day=(20, 35),
        http_ops_per_day=(8, 18),
        device_ops_per_day=(3, 6),
        im_ops_per_day=(10, 18),
    ),
    "vm3": VMConfig(
        vm_id="vm3",
        role="ops_admin",
        primary_levels=["L3", "L4"],
        daily_event_range=(800, 1200),
        event_type_weights={"file": 0.40, "email": 0.20, "http": 0.15, "device": 0.10, "im": 0.15},
        file_ops_per_day=(50, 80),
        email_ops_per_day=(15, 25),
        http_ops_per_day=(10, 20),
        device_ops_per_day=(5, 10),
        im_ops_per_day=(8, 15),
    ),
}

# VM-1 迁移后的配置（行政 → 核心业务）
VM1_MIGRATED_CONFIG = VMConfig(
    vm_id="vm1",
    role="core_business",
    primary_levels=["L3", "L4"],
    daily_event_range=(780, 1150),
    event_type_weights={"file": 0.40, "email": 0.20, "http": 0.15, "device": 0.10, "im": 0.15},
    file_ops_per_day=(45, 70),
    email_ops_per_day=(15, 25),
    http_ops_per_day=(10, 20),
    device_ops_per_day=(5, 10),
    im_ops_per_day=(8, 15),
)


@dataclass
class BehaviorEvent:
    """单条行为事件"""
    event_id: str
    vm_id: str
    user_role: str
    timestamp: str
    event_type: str          # file/email/http/device/im
    action: str              # read/write/send/receive/upload/download/insert/remove/transfer
    target: str              # 文件名/邮件地址/URL/设备名/联系人
    sensitivity_level: str   # L1-L4
    metadata: Dict = field(default_factory=dict)


@dataclass
class OperationSequence:
    """操作序列（由多个事件聚合）"""
    seq_id: str
    vm_id: str
    user_role: str
    channel: str            # email/http/usb/im
    label: str              # normal/anomaly
    difficulty: str         # easy/medium/hard (仅 anomaly)
    events: List[Dict] = field(default_factory=list)
    ground_truth: str = ""
    description: str = ""


# ---- 泄露场景模板（基于 CERT r4.2 的 42 个数据窃取场景） ----

THEFT_SCENARIOS = {
    "email": [
        {"id": "E01", "desc": "直接外发敏感文档附件至私人邮箱", "difficulty": "easy",
         "steps": [("file", "read"), ("email", "send")]},
        {"id": "E02", "desc": "将文档重命名为无害名称后外发", "difficulty": "medium",
         "steps": [("file", "read"), ("file", "write"), ("email", "send")]},
        {"id": "E03", "desc": "分批多次小量邮件外发", "difficulty": "hard",
         "steps": [("file", "read"), ("file", "read"), ("email", "send"), ("email", "send"), ("email", "send")]},
        {"id": "E04", "desc": "转发内部邮件至外部邮箱", "difficulty": "easy",
         "steps": [("email", "receive"), ("email", "send")]},
        {"id": "E05", "desc": "将敏感内容嵌入邮件正文（非附件）", "difficulty": "hard",
         "steps": [("file", "read"), ("email", "send")]},
    ],
    "http": [
        {"id": "H01", "desc": "上传敏感文档至个人云盘", "difficulty": "easy",
         "steps": [("file", "read"), ("http", "upload")]},
        {"id": "H02", "desc": "修改文件扩展名后上传", "difficulty": "medium",
         "steps": [("file", "read"), ("file", "write"), ("http", "upload")]},
        {"id": "H03", "desc": "压缩加密后通过网盘分享链接外发", "difficulty": "hard",
         "steps": [("file", "read"), ("file", "write"), ("http", "upload"), ("email", "send")]},
        {"id": "H04", "desc": "通过代理/VPN上传规避检测", "difficulty": "hard",
         "steps": [("http", "upload")]},
        {"id": "H05", "desc": "利用网页邮箱上传附件", "difficulty": "medium",
         "steps": [("file", "read"), ("http", "upload")]},
    ],
    "usb": [
        {"id": "U01", "desc": "直接拷贝敏感文件至U盘", "difficulty": "easy",
         "steps": [("device", "insert"), ("file", "read"), ("file", "write"), ("device", "remove")]},
        {"id": "U02", "desc": "批量拷贝多份文件后拔出", "difficulty": "medium",
         "steps": [("device", "insert"), ("file", "read"), ("file", "read"), ("file", "write"), ("file", "write"), ("device", "remove")]},
        {"id": "U03", "desc": "非工作时段USB拷贝", "difficulty": "medium",
         "steps": [("device", "insert"), ("file", "read"), ("file", "write"), ("device", "remove")]},
        {"id": "U04", "desc": "USB拷贝后删除本地副本掩盖痕迹", "difficulty": "hard",
         "steps": [("device", "insert"), ("file", "read"), ("file", "write"), ("device", "remove"), ("file", "delete")]},
    ],
    "im": [
        {"id": "M01", "desc": "通过微信直接发送敏感文件", "difficulty": "easy",
         "steps": [("file", "read"), ("im", "transfer")]},
        {"id": "M02", "desc": "截屏后通过IM发送", "difficulty": "medium",
         "steps": [("file", "read"), ("file", "write"), ("im", "transfer")]},
        {"id": "M03", "desc": "将敏感文档拆分为图片通过IM发送", "difficulty": "hard",
         "steps": [("file", "read"), ("file", "write"), ("file", "write"), ("im", "transfer"), ("im", "transfer")]},
        {"id": "M04", "desc": "通过飞书文档分享外传", "difficulty": "medium",
         "steps": [("file", "read"), ("im", "transfer")]},
    ],
}

# ---- 文件名模板 ----
FILE_TEMPLATES = {
    "L1": [
        "company_newsletter_{}.pdf", "training_guide_v{}.docx",
        "public_report_{}.pdf", "team_building_{}.xlsx",
        "office_notice_{}.docx", "product_manual_{}.pdf",
    ],
    "L2": [
        "meeting_minutes_{}.docx", "project_plan_{}.xlsx",
        "internal_memo_{}.pdf", "dept_weekly_{}.docx",
        "test_report_{}.pdf", "design_spec_{}.docx",
    ],
    "L3": [
        "financial_report_Q{}.xlsx", "invoice_{}.pdf",
        "contract_draft_{}.docx", "budget_{}.xlsx",
        "audit_report_{}.pdf", "nda_{}.pdf",
    ],
    "L4": [
        "security_architecture_{}.pdf", "key_management_{}.docx",
        "vuln_assessment_{}.pdf", "salary_plan_{}.xlsx",
        "strategy_roadmap_{}.pdf", "source_code_review_{}.docx",
    ],
}

EMAIL_DOMAINS_INTERNAL = ["company.com", "corp.internal"]
EMAIL_DOMAINS_EXTERNAL = ["gmail.com", "163.com", "qq.com", "outlook.com"]
CLOUD_SERVICES = [
    "pan.baidu.com", "drive.weixin.qq.com", "yunpan.360.cn",
    "cloud.189.cn", "www.jianguoyun.com",
]
USB_DEVICES = ["SanDisk_Cruzer", "Kingston_DT", "Samsung_T7", "Toshiba_USB"]
IM_APPS = ["WeChat.exe", "DingTalk.exe", "WxWork.exe", "Lark.exe"]


class BehaviorSimulator:
    """行为模拟器"""

    def __init__(self, seed: int = 42, start_date: str = "2024-01-01"):
        self.seed = seed
        self.rng = random.Random(seed)
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self._event_counter = 0

    def _gen_event_id(self) -> str:
        self._event_counter += 1
        return f"evt_{self._event_counter:08d}"

    def _gen_timestamp(self, day: int, work_bias: float = 0.85) -> datetime:
        """生成事件时间戳"""
        base = self.start_date + timedelta(days=day)
        if self.rng.random() < work_bias:
            # 工作时段 8:00-18:00
            hour = self.rng.randint(8, 17)
            minute = self.rng.randint(0, 59)
        else:
            # 非工作时段
            hour = self.rng.choice(list(range(0, 8)) + list(range(18, 24)))
            minute = self.rng.randint(0, 59)
        second = self.rng.randint(0, 59)
        return base.replace(hour=hour, minute=minute, second=second)

    def _pick_file(self, levels: List[str]) -> Tuple[str, str]:
        """选择一个文件及其敏感等级"""
        level = self.rng.choice(levels)
        templates = FILE_TEMPLATES[level]
        fname = self.rng.choice(templates).format(self.rng.randint(100, 999))
        return fname, level

    def _gen_normal_event(
        self, vm_config: VMConfig, day: int, event_type: str,
    ) -> BehaviorEvent:
        """生成单条正常行为事件"""
        ts = self._gen_timestamp(day, vm_config.work_hour_bias)
        fname, level = self._pick_file(vm_config.primary_levels)

        if event_type == "file":
            action = self.rng.choice(["read", "write", "read", "read"])  # 读多写少
            target = fname
        elif event_type == "email":
            action = self.rng.choice(["send", "receive", "receive"])
            domain = self.rng.choice(EMAIL_DOMAINS_INTERNAL)
            target = f"user{self.rng.randint(1,50)}@{domain}"
        elif event_type == "http":
            action = self.rng.choice(["download", "upload", "download"])
            target = f"https://internal.company.com/docs/{fname}"
        elif event_type == "device":
            action = self.rng.choice(["insert", "remove"])
            target = self.rng.choice(USB_DEVICES)
        elif event_type == "im":
            action = self.rng.choice(["receive", "transfer", "receive"])
            target = self.rng.choice(IM_APPS)
        else:
            action = "unknown"
            target = ""

        return BehaviorEvent(
            event_id=self._gen_event_id(),
            vm_id=vm_config.vm_id,
            user_role=vm_config.role,
            timestamp=ts.isoformat(),
            event_type=event_type,
            action=action,
            target=target,
            sensitivity_level=level,
        )

    def _gen_anomaly_sequence(
        self, vm_config: VMConfig, channel: str, scenario: Dict, day: int,
    ) -> OperationSequence:
        """生成一条异常操作序列"""
        seq_id = f"seq_anom_{uuid.uuid4().hex[:8]}"
        events = []
        base_ts = self._gen_timestamp(day, 0.5 if scenario["difficulty"] == "medium" else 0.85)

        # 异常序列访问的文件等级偏高
        anomaly_levels = ["L3", "L4"] if self.rng.random() > 0.3 else ["L2", "L3"]

        ts = base_ts
        for i, (etype, action) in enumerate(scenario["steps"]):
            ts += timedelta(seconds=self.rng.randint(5, 120))
            fname, level = self._pick_file(anomaly_levels)

            if etype == "email" and action == "send":
                target = f"personal{self.rng.randint(1,20)}@{self.rng.choice(EMAIL_DOMAINS_EXTERNAL)}"
            elif etype == "http" and action == "upload":
                target = f"https://{self.rng.choice(CLOUD_SERVICES)}/upload"
            elif etype == "device":
                target = self.rng.choice(USB_DEVICES)
            elif etype == "im":
                target = self.rng.choice(IM_APPS)
            else:
                target = fname

            evt = BehaviorEvent(
                event_id=self._gen_event_id(),
                vm_id=vm_config.vm_id,
                user_role=vm_config.role,
                timestamp=ts.isoformat(),
                event_type=etype,
                action=action,
                target=target,
                sensitivity_level=level,
                metadata={"scenario_id": scenario["id"], "step": i},
            )
            events.append(asdict(evt))

        return OperationSequence(
            seq_id=seq_id,
            vm_id=vm_config.vm_id,
            user_role=vm_config.role,
            channel=channel,
            label="anomaly",
            difficulty=scenario["difficulty"],
            events=events,
            ground_truth="data_theft",
            description=scenario["desc"],
        )

    def _gen_normal_sequence(
        self, vm_config: VMConfig, channel: str, day: int,
    ) -> OperationSequence:
        """生成一条正常操作序列"""
        seq_id = f"seq_norm_{uuid.uuid4().hex[:8]}"
        events = []
        base_ts = self._gen_timestamp(day, vm_config.work_hour_bias)

        n_steps = self.rng.randint(2, 5)
        ts = base_ts
        for i in range(n_steps):
            ts += timedelta(seconds=self.rng.randint(10, 300))
            fname, level = self._pick_file(vm_config.primary_levels)

            if channel == "email":
                etype = self.rng.choice(["file", "email"])
                action = "read" if etype == "file" else self.rng.choice(["send", "receive"])
                target = fname if etype == "file" else f"colleague@{self.rng.choice(EMAIL_DOMAINS_INTERNAL)}"
            elif channel == "http":
                etype = self.rng.choice(["file", "http"])
                action = "read" if etype == "file" else "download"
                target = fname if etype == "file" else f"https://internal.company.com/{fname}"
            elif channel == "usb":
                etype = self.rng.choice(["file", "device"])
                action = "read" if etype == "file" else self.rng.choice(["insert", "remove"])
                target = fname if etype == "file" else self.rng.choice(USB_DEVICES)
            elif channel == "im":
                etype = self.rng.choice(["file", "im"])
                action = "read" if etype == "file" else "receive"
                target = fname if etype == "file" else self.rng.choice(IM_APPS)
            else:
                continue

            evt = BehaviorEvent(
                event_id=self._gen_event_id(),
                vm_id=vm_config.vm_id,
                user_role=vm_config.role,
                timestamp=ts.isoformat(),
                event_type=etype,
                action=action,
                target=target,
                sensitivity_level=level,
            )
            events.append(asdict(evt))

        return OperationSequence(
            seq_id=seq_id,
            vm_id=vm_config.vm_id,
            user_role=vm_config.role,
            channel=channel,
            label="normal",
            difficulty="none",
            events=events,
            ground_truth="normal",
            description=f"正常{channel}操作",
        )

    def generate_events(
        self, output_dir: str = "data/behavior_logs",
    ) -> Dict[str, int]:
        """生成 30 天原始行为事件流

        Returns:
            各 VM 的事件计数
        """
        self.rng = random.Random(self.seed)
        self._event_counter = 0
        counts = {}

        for vm_id, vm_config in VM_CONFIGS.items():
            vm_dir = os.path.join(output_dir, vm_id)
            os.makedirs(vm_dir, exist_ok=True)
            vm_events = []

            for day in range(SIMULATION_DAYS):
                # 第 8 天 VM-1 角色迁移
                current_config = vm_config
                if vm_id == "vm1" and day >= ROLE_MIGRATION_DAY:
                    current_config = VM1_MIGRATED_CONFIG

                # 当天事件数
                n_events = self.rng.randint(*current_config.daily_event_range)

                for _ in range(n_events):
                    # 按权重选择事件类型
                    weights = current_config.event_type_weights
                    etype = self.rng.choices(
                        list(weights.keys()),
                        weights=list(weights.values()),
                        k=1,
                    )[0]
                    evt = self._gen_normal_event(current_config, day, etype)
                    vm_events.append(asdict(evt))

            # 按时间排序
            vm_events.sort(key=lambda e: e["timestamp"])

            # 写入 JSONL
            events_path = os.path.join(vm_dir, "events.jsonl")
            with open(events_path, "w", encoding="utf-8") as f:
                for evt in vm_events:
                    f.write(json.dumps(evt, ensure_ascii=False) + "\n")

            counts[vm_id] = len(vm_events)
            print(f"  {vm_id} ({vm_config.role}): {len(vm_events)} 条事件")

        return counts

    def generate_sequences(
        self, output_dir: str = "data/sequences",
    ) -> Dict[str, int]:
        """生成操作序列数据集

        Returns:
            异常/正常序列计数
        """
        self.rng = random.Random(self.seed + 1000)
        self._event_counter = 100000  # 避免与事件 ID 冲突

        os.makedirs(os.path.join(output_dir, "anomaly"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "normal"), exist_ok=True)

        all_anomaly = []
        all_normal = []

        vm_list = list(VM_CONFIGS.values())

        # 生成异常序列
        for channel, quotas in CHANNEL_QUOTAS.items():
            scenarios = THEFT_SCENARIOS.get(channel, [])
            n_anomaly = quotas["anomaly"]

            for i in range(n_anomaly):
                vm_config = self.rng.choice(vm_list)
                day = self.rng.randint(0, SIMULATION_DAYS - 1)

                # 第 8 天后 VM-1 迁移
                if vm_config.vm_id == "vm1" and day >= ROLE_MIGRATION_DAY:
                    vm_config = VM1_MIGRATED_CONFIG

                scenario = self.rng.choice(scenarios)
                seq = self._gen_anomaly_sequence(vm_config, channel, scenario, day)
                all_anomaly.append(seq)

            # 生成正常序列
            n_normal = quotas["normal"]
            for i in range(n_normal):
                vm_config = self.rng.choice(vm_list)
                day = self.rng.randint(0, SIMULATION_DAYS - 1)
                if vm_config.vm_id == "vm1" and day >= ROLE_MIGRATION_DAY:
                    vm_config = VM1_MIGRATED_CONFIG

                seq = self._gen_normal_sequence(vm_config, channel, day)
                all_normal.append(seq)

        # 写入文件
        for seq in all_anomaly:
            path = os.path.join(output_dir, "anomaly", f"{seq.seq_id}.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps(asdict(seq), ensure_ascii=False) + "\n")

        for seq in all_normal:
            path = os.path.join(output_dir, "normal", f"{seq.seq_id}.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps(asdict(seq), ensure_ascii=False) + "\n")

        # 写入汇总索引
        summary = {
            "total_anomaly": len(all_anomaly),
            "total_normal": len(all_normal),
            "by_channel": {},
            "by_difficulty": {},
        }
        for seq in all_anomaly:
            ch = seq.channel
            summary["by_channel"][ch] = summary["by_channel"].get(ch, {"anomaly": 0, "normal": 0})
            summary["by_channel"][ch]["anomaly"] += 1
            d = seq.difficulty
            summary["by_difficulty"][d] = summary["by_difficulty"].get(d, 0) + 1
        for seq in all_normal:
            ch = seq.channel
            summary["by_channel"][ch] = summary["by_channel"].get(ch, {"anomaly": 0, "normal": 0})
            summary["by_channel"][ch]["normal"] += 1

        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return {"anomaly": len(all_anomaly), "normal": len(all_normal)}

    def run(
        self,
        events_dir: str = "data/behavior_logs",
        sequences_dir: str = "data/sequences",
    ) -> Dict:
        """运行完整模拟"""
        print("[行为模拟器] 开始生成 30 天行为数据...")
        print(f"  随机种子: {self.seed}")
        print(f"  起始日期: {self.start_date.strftime('%Y-%m-%d')}")
        print(f"  角色迁移: 第 {ROLE_MIGRATION_DAY} 天 VM-1 行政→核心业务")
        print()

        print("[1/2] 生成原始行为事件流:")
        event_counts = self.generate_events(events_dir)
        total_events = sum(event_counts.values())
        print(f"  合计: {total_events} 条事件")
        print()

        print("[2/2] 生成操作序列数据集:")
        seq_counts = self.generate_sequences(sequences_dir)
        print(f"  异常序列: {seq_counts['anomaly']} 条")
        print(f"  正常序列: {seq_counts['normal']} 条")
        print(f"  合计: {seq_counts['anomaly'] + seq_counts['normal']} 条")

        return {
            "events": event_counts,
            "total_events": total_events,
            "sequences": seq_counts,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CERT r4.2 行为模拟器")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--events-dir",
        default="data/behavior_logs",
        help="事件输出目录",
    )
    parser.add_argument(
        "--sequences-dir",
        default="data/sequences",
        help="序列输出目录",
    )
    parser.add_argument(
        "--start-date", default="2024-01-01",
        help="模拟起始日期 (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    simulator = BehaviorSimulator(seed=args.seed, start_date=args.start_date)
    result = simulator.run(
        events_dir=args.events_dir,
        sequences_dir=args.sequences_dir,
    )
    print(f"\n完成: {result}")
