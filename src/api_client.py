"""
DLP 检测系统 — 终端代理远程客户端

部署在终端侧，负责：
1. 向服务端注册 Agent
2. 维持心跳（保活 + 拉取规则）
3. 上报文件事件到服务端（HTTP 或 WebSocket）
4. 接收服务端推送的 VQL 规则

使用方式：
    from src.api_client import RemoteClient

    client = RemoteClient(
        server_url="http://192.168.1.100:8900",
        agent_id="agent_vm2",
        user_id="vm2_bob",
    )
    client.register()
    client.start_heartbeat()

    # 作为 TerminalAgent 的 event_callback
    agent.set_event_callback(client.report_event)
"""

import base64
import logging
import json
import threading
import time
from typing import Callable, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class RemoteClient:
    """终端代理远程客户端 — 与 DLP API 服务端通信"""

    def __init__(
        self,
        server_url: str,
        agent_id: str,
        user_id: str,
        api_token: str = "",
        heartbeat_interval: float = 10.0,
        timeout: float = 30.0,
    ):
        self.server_url = server_url.rstrip("/")
        self.agent_id = agent_id
        self.user_id = user_id
        self.api_token = api_token
        self.heartbeat_interval = heartbeat_interval
        self.timeout = timeout

        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

        self._registered = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._running = False

        # 回调：当收到服务端规则推送时调用
        self._rule_callback: Optional[Callable] = None

        # 统计
        self._stats = {
            "events_sent": 0,
            "events_failed": 0,
            "rules_received": 0,
        }

    # -----------------------------------------------------------------
    # 注册
    # -----------------------------------------------------------------

    def register(
        self,
        hostname: str = "",
        os_info: str = "",
        watch_dirs: List[str] = None,
    ) -> bool:
        """向服务端注册 Agent"""
        url = f"{self.server_url}/api/v1/agents/register"
        payload = {
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "hostname": hostname,
            "os_info": os_info,
            "watch_dirs": watch_dirs or [],
            "api_token": self.api_token,
        }

        try:
            resp = self._session.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            self._registered = data.get("success", False)
            if self._registered:
                logger.info(f"[Client] Registered: {self.agent_id}")
            return self._registered
        except Exception as e:
            logger.error(f"[Client] Register failed: {e}")
            return False

    # -----------------------------------------------------------------
    # 心跳 + 规则拉取
    # -----------------------------------------------------------------

    def start_heartbeat(self):
        """启动心跳线程（同时定期拉取规则）"""
        if self._running:
            return
        self._running = True
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="client-heartbeat"
        )
        self._heartbeat_thread.start()

    def stop(self):
        """停止客户端"""
        self._running = False
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)
        self._session.close()

    def _heartbeat_loop(self):
        while self._running:
            try:
                self._send_heartbeat()
                self._poll_rules()
            except Exception as e:
                logger.debug(f"[Client] Heartbeat error: {e}")
            time.sleep(self.heartbeat_interval)

    def _send_heartbeat(self):
        url = f"{self.server_url}/api/v1/agents/heartbeat"
        payload = {
            "agent_id": self.agent_id,
            "uptime_seconds": 0,
            "events_processed": self._stats["events_sent"],
            "active_rules": 0,
        }
        resp = self._session.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()

    def _poll_rules(self):
        """拉取待下发规则"""
        url = f"{self.server_url}/api/v1/rules/{self.agent_id}"
        resp = self._session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        rules = data.get("rules", [])
        for rule in rules:
            self._stats["rules_received"] += 1
            if self._rule_callback:
                self._rule_callback(rule["rule_id"], rule["vql_script"])

    # -----------------------------------------------------------------
    # 事件上报
    # -----------------------------------------------------------------

    def report_event(
        self,
        file_path: str,
        file_content: bytes,
        user_id: str,
        action: str = "read",
        process_name: str = "explorer.exe",
        device_type: str = "local",
        network_env: str = "internal",
        is_work_hours: bool = True,
        **kwargs,
    ) -> Optional[Dict]:
        """上报文件事件到服务端（同步 HTTP）

        签名与 DLPServer.process_file_event 兼容，
        可直接替代本地回调：
            agent.set_event_callback(client.report_event)

        Returns:
            服务端返回的 VerdictResponse 字典，或 None（上报失败）
        """
        url = f"{self.server_url}/api/v1/events"
        payload = {
            "agent_id": self.agent_id,
            "file_path": file_path,
            "file_content_b64": base64.b64encode(file_content).decode("ascii"),
            "user_id": user_id,
            "action": action,
            "process_name": process_name,
            "device_type": device_type,
            "network_env": network_env,
            "is_work_hours": is_work_hours,
            "metadata": kwargs.get("metadata", {}),
        }

        try:
            resp = self._session.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            self._stats["events_sent"] += 1
            return resp.json()
        except Exception as e:
            self._stats["events_failed"] += 1
            logger.error(f"[Client] Event report failed: {e}")
            return None

    # -----------------------------------------------------------------
    # 规则接收回调
    # -----------------------------------------------------------------

    def set_rule_callback(self, callback: Callable):
        """设置规则接收回调

        callback 签名: (rule_id: str, vql_script: str) -> None
        对应 VQLExecutor.load_rule
        """
        self._rule_callback = callback

    # -----------------------------------------------------------------
    # WebSocket 模式（可选增强）
    # -----------------------------------------------------------------

    def connect_ws(self) -> bool:
        """建立 WebSocket 长连接（需要 websockets 库）

        WebSocket 模式下：
        - 规则由服务端主动推送（无需轮询）
        - 事件可通过 WS 上报（更低延迟）
        """
        try:
            import websockets
            import asyncio
        except ImportError:
            logger.warning("[Client] websockets not installed, falling back to HTTP polling")
            return False

        self._ws_url = f"{self.server_url.replace('http', 'ws')}/api/v1/ws/{self.agent_id}"
        self._ws_thread = threading.Thread(
            target=self._ws_loop, daemon=True, name="client-ws"
        )
        self._ws_thread.start()
        return True

    def _ws_loop(self):
        """WebSocket 事件循环"""
        import asyncio

        async def _run():
            try:
                import websockets
            except ImportError:
                return

            while self._running:
                try:
                    async with websockets.connect(self._ws_url) as ws:
                        logger.info(f"[Client] WebSocket connected: {self._ws_url}")
                        while self._running:
                            try:
                                msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
                                data = json.loads(msg)
                                self._handle_ws_message(data)
                            except asyncio.TimeoutError:
                                # 发送心跳
                                await ws.send(json.dumps({"type": "heartbeat"}))
                except Exception as e:
                    logger.debug(f"[Client] WS error: {e}, reconnecting in 5s...")
                    await asyncio.sleep(5)

        asyncio.run(_run())

    def _handle_ws_message(self, data: Dict):
        """处理 WebSocket 消息"""
        msg_type = data.get("type", "")
        if msg_type == "rule_push":
            rule = data.get("data", {})
            self._stats["rules_received"] += 1
            if self._rule_callback:
                self._rule_callback(rule.get("rule_id", ""), rule.get("vql_script", ""))
        elif msg_type == "verdict":
            # 异步事件上报的响应
            pass

    # -----------------------------------------------------------------
    # 状态
    # -----------------------------------------------------------------

    @property
    def is_registered(self) -> bool:
        return self._registered

    def get_stats(self) -> Dict:
        return dict(self._stats)
