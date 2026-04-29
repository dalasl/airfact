#!/usr/bin/env python3
"""
Terminal Data Leakage Detection System
Based on User Data Feature Profiling

Entry point for deployment. Start the detection server:

    python main.py serve                             # start full system
    python main.py serve --config configs/xxx.yaml   # custom config
    python main.py serve --device cpu                # CPU-only mode
    python main.py scan  report.pdf --user alice     # scan single file
    python main.py demo                              # demo with sample events
    python main.py report                            # view experiment results
"""

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("dlp")


# =====================================================================
# 用户与测试数据（来自 CERT r4.2 LDAP 员工岗位体系）
# =====================================================================

USERS = [
    {
        "user_id": "vm1_alice",   "role": "office_staff",
        "department": "marketing",  "projects": ["branding"],
        "permissions": ["doc_rw", "email"],
        "business_keywords": ["market_analysis", "competitor_report"],
    },
    {
        "user_id": "vm2_bob",     "role": "finance",
        "department": "finance",    "projects": ["Q1_report", "budget"],
        "permissions": ["finance_system", "doc_rw", "email"],
        "business_keywords": ["balance_sheet", "profit_loss", "audit"],
    },
    {
        "user_id": "vm3_charlie", "role": "ops_admin",
        "department": "infosec",    "projects": ["infra", "sec_ops", "key_mgmt"],
        "permissions": ["full_access", "server_admin", "db_admin"],
        "business_keywords": ["server_config", "key_rotation", "AES", "RSA"],
    },
]

SEED_CASES = [
    {"doc_summary": "public annual report",    "level": "L1", "reason": "published"},
    {"doc_summary": "internal meeting minutes", "level": "L2", "reason": "internal only"},
    {"doc_summary": "customer contract",        "level": "L3", "reason": "business sensitive"},
    {"doc_summary": "algorithm source code",    "level": "L4", "reason": "core IP"},
    {"doc_summary": "salary spreadsheet",       "level": "L4", "reason": "PII"},
    {"doc_summary": "DB root password list",    "level": "L4", "reason": "top secret credential"},
]

TEST_EVENTS = [
    {
        "tag": "email/public/pass",
        "file_path": "C:/Users/alice/Documents/brochure_2026.pdf",
        "file_content": b"Brand brochure 2026 - public release",
        "user_id": "vm1_alice",  "action": "send",
        "process_name": "outlook.exe",
        "device_type": "local",  "network_env": "internal", "is_work_hours": True,
    },
    {
        "tag": "usb/finance/alert",
        "file_path": "C:/Users/bob/Documents/Q1_financial_report.xlsx",
        "file_content": b"Q1 Financial Statement - revenue 12.5M",
        "user_id": "vm2_bob",    "action": "copy",
        "process_name": "explorer.exe",
        "device_type": "usb",    "network_env": "internal", "is_work_hours": True,
    },
    {
        "tag": "http/credential/block",
        "file_path": "C:/Admin/Keys/server_root_credentials.txt",
        "file_content": b"root password: P@ssw0rd!2026\nDB master key: AES-256-CBC",
        "user_id": "vm3_charlie","action": "upload",
        "process_name": "chrome.exe",
        "device_type": "local",  "network_env": "external", "is_work_hours": False,
    },
    {
        "tag": "im/contract/alert",
        "file_path": "C:/Users/alice/Documents/contract_ABC.docx",
        "file_content": b"Contract with ABC Corp, total value 2.5M",
        "user_id": "vm1_alice",  "action": "send",
        "process_name": "WeChat.exe",
        "device_type": "local",  "network_env": "external", "is_work_hours": True,
    },
    {
        "tag": "local/config/pass",
        "file_path": "C:/Admin/Configs/nginx.conf",
        "file_content": b"server { listen 80; server_name internal.company.com; }",
        "user_id": "vm3_charlie","action": "read",
        "process_name": "notepad++.exe",
        "device_type": "local",  "network_env": "internal", "is_work_hours": True,
    },
]


# =====================================================================
# serve — 启动完整检测系统
# =====================================================================

def cmd_serve(args):
    """Start the full detection system.

    Initialization sequence:
        1. Load config & create DLPServer
        2. Register users from LDAP / config
        3. Init PerceptionService  (ch2: clustering, profile baselines)
        4. Init CognitionService   (ch3: RAG seed injection)
        5. Init ExecutionService   (ch4: threshold calibration)
        6. Connect VelociraptorBridge (gRPC -> Go VQL engine)
        7. Start TerminalAgent     (file watcher + process monitor)
        8. Enter main event loop   (route events through three layers)
    """
    from src.server import DLPServer
    from src.terminal_agent import TerminalAgent, ProcessWatcher
    from src.event_bus import EventType
    import numpy as np

    np.random.seed(42)
    config_path = args.config

    # ── Step 1: Create server ──
    log.info("=" * 60)
    log.info("Terminal Data Leakage Detection System")
    log.info("=" * 60)
    log.info("[1/8] Loading config: %s", config_path)
    server = DLPServer.from_config(config_path, device=args.device)

    # ── Step 2-5: Init three-layer services ──
    log.info("[2/8] Registering %d users", len(USERS))
    historical_features = np.random.randn(len(USERS), 896).astype(np.float32)
    server.init(
        users_meta=USERS,
        seed_cases=SEED_CASES,
        historical_features=historical_features,
    )
    log.info("[3/8] PerceptionService ready  (ch2: IGW-Kmeans K=%d)",
             server.perception.igw_kmeans.K)
    log.info("[4/8] CognitionService ready   (ch3: %d RAG seeds)",
             len(SEED_CASES))
    log.info("[5/8] ExecutionService ready    (ch4: thresholds calibrated)")

    # ── Step 6: Connect Velociraptor ──
    log.info("[6/8] Connecting VelociraptorBridge...")
    velo_status = "offline"
    try:
        from src.velociraptor_bridge import VelociraptorBridge, VelociraptorConfig
        bridge = VelociraptorBridge(VelociraptorConfig())
        if bridge.connect():
            server.velo_bridge = bridge
            velo_status = "online"
            log.info("       Velociraptor Server connected (gRPC)")
        else:
            log.info("       Velociraptor offline — VQL rules local-only")
    except ImportError:
        log.info("       pyvelociraptor not installed — VQL rules local-only")
    log.info("       Source: velociraptor-master/actions/vql.go")

    # ── Step 7: Start TerminalAgent ──
    log.info("[7/8] Starting TerminalAgent...")
    watch_dirs = []
    if args.watch:
        watch_dirs = [args.watch]
    agent = TerminalAgent(
        user_id=USERS[0]["user_id"],
        watch_dirs=watch_dirs,
        poll_interval=args.interval,
    )
    # Wire agent -> server: when agent detects sensitive file, send to server
    agent.set_event_callback(server.process_file_event)

    # Wire server -> agent: when server generates VQL rule, push to agent
    def on_rule_pushed(event):
        vql = event.data.get("vql_script", "")
        rid = event.data.get("rule_id", "")
        if vql:
            agent.vql_executor.load_rule(rid, vql)
            log.info("       Rule pushed to agent: %s", rid)

    server.event_bus.subscribe(
        EventType.RULE_PUSHED, on_rule_pushed, "terminal_agent"
    )

    if watch_dirs:
        agent.start()
        log.info("       Watching: %s", watch_dirs)
    else:
        log.info("       No --watch dir, agent standby")

    # ── Step 7b: Subscribe Velociraptor client events (上行通道) ──
    velo_listener = None
    shutdown_event = threading.Event()
    if velo_status == "online":

        def _on_velo_event(event_dict):
            """将 Velociraptor 终端回传事件送入三层流水线"""
            file_path = event_dict.get("FullPath", event_dict.get("Path", ""))
            client_id = event_dict.get("ClientId", "unknown")
            # 根据 ClientId 映射到用户（VM-1/2/3）
            user_map = {uid: uid for uid in [u["user_id"] for u in USERS]}
            user_id = event_dict.get("UserID", "")
            if not user_id:
                # 从已注册用户中按 client_id 查找
                for u in USERS:
                    if client_id in u.get("client_ids", [client_id]):
                        user_id = u["user_id"]
                        break
                else:
                    user_id = USERS[0]["user_id"]

            # 读取文件内容（远程事件可能无本地文件，使用元数据）
            file_content = event_dict.get("Content", b"")
            if not file_content and os.path.exists(file_path):
                try:
                    with open(file_path, "rb") as f:
                        file_content = f.read(8192)
                except OSError:
                    file_content = file_path.encode()

            action = event_dict.get("Action", "read")
            process_name = event_dict.get("ProcessName", "velociraptor.exe")
            device_type = event_dict.get("DeviceType", "local")
            network_env = event_dict.get("NetworkEnv", "internal")

            verdict = server.process_file_event(
                file_path=file_path,
                file_content=file_content if isinstance(file_content, bytes)
                    else file_content.encode(),
                user_id=user_id,
                action=action,
                process_name=process_name,
                device_type=device_type,
                network_env=network_env,
            )
            log.info(
                "  [VELO] %s | %s | %s | risk=%.3f | %s",
                client_id, file_path, verdict.sensitivity_level,
                verdict.risk_score, verdict.response_action,
            )

        def _velo_listener_thread():
            log.info("       Velociraptor event listener started")
            bridge.subscribe_client_events(
                artifact_prefix="Custom.DLP.",
                callback=_on_velo_event,
                stop_event=shutdown_event,
            )
            log.info("       Velociraptor event listener stopped")

        velo_listener = threading.Thread(
            target=_velo_listener_thread, daemon=True, name="velo-listener"
        )
        velo_listener.start()
        log.info("       Velociraptor event listener: active")

    # ── Step 8: Main loop ──
    log.info("[8/8] System running")
    log.info("-" * 60)
    log.info("  PerceptionService  -->  ProfileMatchResult")
    log.info("       |")
    log.info("       v")
    log.info("  CognitionService   -->  GradingResult")
    log.info("       |")
    log.info("       v")
    log.info("  ExecutionService   -->  DetectionVerdict")
    log.info("       |                  + RuleDeployment")
    log.info("       v")
    log.info("  VelociraptorBridge -->  Artifact (gRPC -> Go)")
    log.info("  EventBus           -->  broadcast to subscribers")
    log.info("-" * 60)
    log.info("  Velociraptor : %s", velo_status)
    log.info("  Device       : %s", args.device)
    log.info("  Config       : %s", config_path)
    if watch_dirs:
        log.info("  Watch dir    : %s", watch_dirs[0])
    log.info("  PID          : %d", os.getpid())
    log.info("")
    log.info("System ready. Waiting for file events... (Ctrl+C to stop)")

    # Graceful shutdown
    shutdown = threading.Event()

    def handle_signal(signum, frame):
        log.info("Received signal %d, shutting down...", signum)
        shutdown.set()

    signal.signal(signal.SIGINT,  handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        while not shutdown.is_set():
            shutdown.wait(timeout=1.0)
    except KeyboardInterrupt:
        pass
    finally:
        log.info("Shutting down...")
        # 停止 Velociraptor 事件监听
        if velo_listener and velo_listener.is_alive():
            shutdown_event.set()
            velo_listener.join(timeout=3)
            log.info("  Velociraptor listener stopped")
        agent.stop()
        if server.velo_bridge:
            server.velo_bridge.close()
        stats = server.get_system_stats()
        log.info("Final stats: %s", json.dumps(stats, default=str))
        log.info("Shutdown complete.")


# =====================================================================
# demo — 演示完整检测流程
# =====================================================================

def cmd_demo(args):
    """Run demo with sample events showing three-layer collaboration."""
    from src.server import DLPServer
    from src.terminal_agent import ProcessWatcher, VQLExecutor
    import numpy as np

    np.random.seed(42)
    config_path = args.config

    print("=" * 65)
    print("  Terminal Data Leakage Detection System — Demo")
    print("=" * 65)

    # ── Init ──
    print("\n[init] Loading config:", config_path)
    server = DLPServer.from_config(config_path, device="cpu")
    features = np.random.randn(len(USERS), 896).astype(np.float32)
    server.init(users_meta=USERS, seed_cases=SEED_CASES,
                historical_features=features)

    for u in USERS:
        print(f"  user {u['user_id']:15s}  role={u['role']:12s}  dept={u['department']}")

    # ── Velociraptor status ──
    print("\n[velociraptor] source: velociraptor-master/")
    print("  actions/vql.go       VQLClientAction.StartQuery()")
    print("  api/proto/api.proto  rpc Query(VQLCollectorArgs) -> stream VQLResponse")
    print("  accessors/           NTFS, registry, S3, SMB, process memory, ...")
    print("  artifacts/           54 built-in Artifact YAML definitions")

    # ── Process events ──
    print("\n" + "-" * 65)
    print("[detection] Processing %d events through three-layer services" % len(TEST_EVENTS))
    print("  PerceptionService -> ProfileMatchResult -> CognitionService")
    print("  CognitionService  -> GradingResult      -> ExecutionService")
    print("  ExecutionService  -> DetectionVerdict    -> EventBus broadcast")
    print("  ExecutionService  -> RuleDeployment      -> VelociraptorBridge")
    print("-" * 65)

    verdicts = []
    for i, ev in enumerate(TEST_EVENTS, 1):
        tag = ev.pop("tag")
        verdict = server.process_file_event(**ev)
        verdicts.append(verdict)
        ev["tag"] = tag

        action_sym = {"block": "BLOCK", "alert": "ALERT", "silent_monitoring": "PASS "}
        sym = action_sym.get(verdict.response_action, "?????")
        print(f"\n  [{sym}] event {i}: {tag}")
        print(f"         file  = {ev['file_path']}")
        print(f"         user  = {ev['user_id']}  process = {ev['process_name']}")
        print(f"         level = {verdict.sensitivity_level}  "
              f"risk = {verdict.risk_score:.3f}  "
              f"conf = {verdict.grading_confidence:.2f}")
        print(f"         s_u = {verdict.subject_consistency:.3f}  "
              f"d_b = {verdict.behavior_deviation:.3f}  "
              f"c_e = {verdict.env_confidence:.3f}")
        if verdict.vql_script:
            print(f"         vql  = {verdict.vql_script[:70]}...")
        print(f"         time = {verdict.latency_ms:.0f} ms")

    # ── Terminal agent demo ──
    print("\n" + "-" * 65)
    print("[agent] Process -> channel identification:")
    pw = ProcessWatcher()
    for proc, dev in [("outlook.exe","local"),("WeChat.exe","local"),
                      ("chrome.exe","local"),("explorer.exe","usb")]:
        ch = pw.identify_channel(proc, dev)
        print(f"  {proc:18s} + {dev:5s}  ->  {ch.value}")

    vql_exec = VQLExecutor()
    rules = [v for v in verdicts if v.vql_script]
    if rules:
        print(f"\n[agent] Loading {len(rules)} VQL rules into local executor:")
        for i, v in enumerate(rules):
            ok = vql_exec.load_rule(f"rule_{i}", v.vql_script)
            print(f"  {'OK' if ok else 'FAIL'}  rule_{i}: {v.vql_script[:55]}...")

    # ── Summary ──
    print("\n" + "-" * 65)
    print("[summary]")
    blocked = sum(1 for v in verdicts if v.response_action == "block")
    alerted = sum(1 for v in verdicts if v.response_action == "alert")
    passed  = sum(1 for v in verdicts if v.response_action == "silent_monitoring")
    avg_ms  = sum(v.latency_ms for v in verdicts) / len(verdicts)
    print(f"  events    = {len(verdicts)}")
    print(f"  blocked   = {blocked}")
    print(f"  alerted   = {alerted}")
    print(f"  passed    = {passed}")
    print(f"  avg time  = {avg_ms:.0f} ms")
    print(f"\n  event bus stats: {server.event_bus.get_stats()['event_counts']}")
    print("=" * 65)


# =====================================================================
# scan — 扫描单个文件
# =====================================================================

def cmd_scan(args):
    """Scan a single file and output detection verdict."""
    from src.server import DLPServer

    if not os.path.exists(args.file):
        log.error("File not found: %s", args.file)
        sys.exit(1)

    server = DLPServer.from_config(
        args.config or "configs/default_config.yaml", device="cpu",
    )
    server.init(users_meta=[{
        "user_id": args.user, "role": args.role or "default",
        "department": "", "projects": [], "permissions": [],
        "business_keywords": [],
    }], seed_cases=SEED_CASES)

    with open(args.file, "rb") as f:
        content = f.read()

    verdict = server.process_file_event(
        file_path=args.file, file_content=content,
        user_id=args.user, action=args.action or "read",
        process_name="cli_scan",
    )

    print(verdict.summary())
    print(f"  reason: {verdict.reason}")

    if args.json:
        import dataclasses
        print(json.dumps(dataclasses.asdict(verdict), default=str, indent=2))


# =====================================================================
# monitor — 持续监控目录
# =====================================================================

def cmd_monitor(args):
    """Monitor a directory for file changes."""
    from src.server import DLPServer
    from src.terminal_agent import TerminalAgent

    if not os.path.isdir(args.watch):
        log.error("Directory not found: %s", args.watch)
        sys.exit(1)

    server = DLPServer.from_config(
        args.config or "configs/default_config.yaml", device="cpu",
    )
    server.init(users_meta=[{
        "user_id": args.user, "role": "default",
        "department": "", "projects": [], "permissions": [],
        "business_keywords": [],
    }], seed_cases=SEED_CASES)

    agent = TerminalAgent(
        user_id=args.user,
        watch_dirs=[args.watch],
        poll_interval=args.interval,
    )
    agent.set_event_callback(server.process_file_event)
    agent.start()

    log.info("Monitoring %s (interval=%ds, Ctrl+C to stop)", args.watch, args.interval)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        agent.stop()
        log.info("Stopped.")


# =====================================================================
# report — 查看实验结果
# =====================================================================

def cmd_report(args):
    """Display experiment results summary."""
    results_dir = "results/"
    import glob as g

    print("=" * 60)
    print("  Experiment Results Summary")
    print("=" * 60)

    for f in sorted(g.glob(os.path.join(results_dir, "*.csv"))):
        name = os.path.basename(f).replace(".csv", "")
        with open(f, "r") as fh:
            lines = fh.readlines()
        print(f"\n  {name}")
        print(f"    columns : {lines[0].strip()}")
        print(f"    rows    : {len(lines) - 1}")

    summary_path = os.path.join(results_dir, "experiment_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"\n  Key Results:")
        for k, v in data.get("key_results", {}).items():
            print(f"    {k}: {v}")


# =====================================================================
# CLI entry point
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        prog="dlp-detection",
        description="Terminal Data Leakage Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python main.py serve                               # start detection server
  python main.py serve --watch C:/Users/bob/Documents # with file monitoring
  python main.py serve --device cpu                   # CPU-only mode
  python main.py demo                                 # demo with sample events
  python main.py scan  report.pdf --user bob          # scan single file
  python main.py monitor --watch ./data/              # lightweight monitor
  python main.py report                               # view experiment results
        """,
    )
    sub = parser.add_subparsers(dest="command")

    # --- serve (primary) ---
    p_serve = sub.add_parser("serve", help="Start the full detection system")
    p_serve.add_argument("--config", default="configs/default_config.yaml")
    p_serve.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p_serve.add_argument("--watch",  default=None, help="Directory to monitor")
    p_serve.add_argument("--interval", type=int, default=5, help="Poll interval (sec)")

    # --- demo ---
    p_demo = sub.add_parser("demo", help="Run demo with sample events")
    p_demo.add_argument("--config", default="configs/default_config.yaml")

    # --- scan ---
    p_scan = sub.add_parser("scan", help="Scan a single file")
    p_scan.add_argument("file", help="File path to scan")
    p_scan.add_argument("--user", default="default_user")
    p_scan.add_argument("--role", default=None)
    p_scan.add_argument("--action", default="read")
    p_scan.add_argument("--config", default=None)
    p_scan.add_argument("--json", action="store_true")

    # --- monitor ---
    p_mon = sub.add_parser("monitor", help="Monitor directory for changes")
    p_mon.add_argument("--watch", required=True, help="Directory to watch")
    p_mon.add_argument("--user", default="monitor_user")
    p_mon.add_argument("--interval", type=int, default=5)
    p_mon.add_argument("--config", default=None)

    # --- report ---
    sub.add_parser("report", help="View experiment results")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        print("\nRun 'python main.py serve' to start the detection system")
        print("Run 'python main.py demo'  to see a demo with sample events")
        sys.exit(0)

    dispatch = {
        "serve":   cmd_serve,
        "demo":    cmd_demo,
        "scan":    cmd_scan,
        "monitor": cmd_monitor,
        "report":  cmd_report,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
