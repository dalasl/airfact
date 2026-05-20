"""
DLP 检测系统 — FastAPI 服务端 API 层

在 DLPServer 之上提供网络接口，实现论文图 4-3 所描述的
服务端(GPU服务器) + 多终端代理(Windows VM) 分离部署架构。

通信协议：
    - HTTP REST: 事件上报、Agent 注册/心跳、规则拉取
    - WebSocket: 规则实时推送、双向通信

部署模式：
    python main.py serve-api --host 0.0.0.0 --port 8900

终端代理连接：
    agent = TerminalAgent(...)
    client = RemoteClient(server_url="http://<server>:8900")
    agent.set_event_callback(client.report_event)
"""

import asyncio
import base64
import hashlib
import logging
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.event_bus import EventType, Event
from src.server import DLPServer, DetectionVerdict, RuleDeployment

logger = logging.getLogger(__name__)


# =====================================================================
# Pydantic 模型 — API 请求/响应数据结构
# =====================================================================

class AgentRegisterRequest(BaseModel):
    """终端代理注册请求"""
    agent_id: str = Field(..., description="代理唯一标识")
    user_id: str = Field(..., description="代理关联的用户ID")
    hostname: str = Field(default="", description="主机名")
    os_info: str = Field(default="", description="操作系统信息")
    watch_dirs: List[str] = Field(default_factory=list, description="监控目录列表")
    api_token: str = Field(default="", description="认证 Token")


class AgentRegisterResponse(BaseModel):
    """注册响应"""
    success: bool
    agent_id: str
    message: str = ""
    server_time: float = Field(default_factory=time.time)


class HeartbeatRequest(BaseModel):
    """心跳请求"""
    agent_id: str
    uptime_seconds: float = 0
    events_processed: int = 0
    active_rules: int = 0
    cpu_percent: float = 0
    memory_mb: float = 0


class HeartbeatResponse(BaseModel):
    """心跳响应"""
    ok: bool = True
    server_time: float = Field(default_factory=time.time)
    pending_rules: int = 0


class FileEventRequest(BaseModel):
    """文件事件上报请求"""
    agent_id: str
    file_path: str
    file_content_b64: str = Field(..., description="文件内容 Base64 编码")
    user_id: str
    action: str = "read"
    process_name: str = "explorer.exe"
    device_type: str = "local"
    network_env: str = "internal"
    is_work_hours: bool = True
    file_hash: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VerdictResponse(BaseModel):
    """检测裁决响应"""
    response_action: str
    risk_score: float
    sensitivity_level: str
    grading_confidence: float
    user_id: str
    file_path: str
    vql_script: str = ""
    reason: str = ""
    latency_ms: float = 0
    correlation_id: str = ""


class RuleItem(BaseModel):
    """单条规则"""
    rule_id: str
    vql_script: str
    target_role: str = ""
    sensitivity_level: str = ""
    response_action: str = "alert"


class PendingRulesResponse(BaseModel):
    """待下发规则列表"""
    agent_id: str
    rules: List[RuleItem] = Field(default_factory=list)
    server_time: float = Field(default_factory=time.time)


class SystemStatusResponse(BaseModel):
    """系统状态"""
    users: int = 0
    initialized: bool = False
    velociraptor_connected: bool = False
    connected_agents: int = 0
    total_events_processed: int = 0
    uptime_seconds: float = 0


# =====================================================================
# Agent 连接管理
# =====================================================================

@dataclass
class AgentConnection:
    """已注册的 Agent 连接信息"""
    agent_id: str
    user_id: str
    hostname: str = ""
    os_info: str = ""
    watch_dirs: List[str] = field(default_factory=list)
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    websocket: Optional[WebSocket] = None
    pending_rules: List[Dict] = field(default_factory=list)
    events_processed: int = 0


class AgentManager:
    """管理所有已连接的终端代理"""

    def __init__(self):
        self._agents: Dict[str, AgentConnection] = {}
        self._lock = asyncio.Lock()

    async def register(self, req: AgentRegisterRequest) -> AgentConnection:
        async with self._lock:
            conn = AgentConnection(
                agent_id=req.agent_id,
                user_id=req.user_id,
                hostname=req.hostname,
                os_info=req.os_info,
                watch_dirs=req.watch_dirs,
            )
            self._agents[req.agent_id] = conn
            return conn

    async def heartbeat(self, agent_id: str) -> Optional[AgentConnection]:
        async with self._lock:
            conn = self._agents.get(agent_id)
            if conn:
                conn.last_heartbeat = time.time()
            return conn

    async def get(self, agent_id: str) -> Optional[AgentConnection]:
        return self._agents.get(agent_id)

    async def set_websocket(self, agent_id: str, ws: WebSocket):
        async with self._lock:
            conn = self._agents.get(agent_id)
            if conn:
                conn.websocket = ws

    async def remove_websocket(self, agent_id: str):
        async with self._lock:
            conn = self._agents.get(agent_id)
            if conn:
                conn.websocket = None

    async def push_rule(self, agent_id: str, rule: Dict):
        """推送规则到指定 Agent（优先 WebSocket，否则存入待拉取队列）"""
        conn = self._agents.get(agent_id)
        if not conn:
            return

        # 尝试 WebSocket 推送
        if conn.websocket:
            try:
                await conn.websocket.send_json({
                    "type": "rule_push",
                    "data": rule,
                })
                return
            except Exception:
                conn.websocket = None

        # 回退：存入待拉取队列
        conn.pending_rules.append(rule)

    async def push_rule_to_role(self, target_role: str, rule: Dict):
        """按角色推送规则到所有匹配的 Agent"""
        for conn in self._agents.values():
            # 通过用户角色匹配（需要服务端持有用户→角色映射）
            await self.push_rule(conn.agent_id, rule)

    async def pop_pending_rules(self, agent_id: str) -> List[Dict]:
        async with self._lock:
            conn = self._agents.get(agent_id)
            if conn and conn.pending_rules:
                rules = conn.pending_rules[:]
                conn.pending_rules.clear()
                return rules
            return []

    @property
    def connected_count(self) -> int:
        return len(self._agents)

    def get_all_agents_status(self) -> List[Dict]:
        now = time.time()
        return [
            {
                "agent_id": c.agent_id,
                "user_id": c.user_id,
                "hostname": c.hostname,
                "online": (now - c.last_heartbeat) < 30,
                "ws_connected": c.websocket is not None,
                "pending_rules": len(c.pending_rules),
                "events_processed": c.events_processed,
            }
            for c in self._agents.values()
        ]


# =====================================================================
# FastAPI 应用
# =====================================================================

class DLPAPIServer:
    """DLP 检测系统 API 服务封装"""

    def __init__(self, dlp_server: DLPServer, api_token: str = ""):
        self.dlp_server = dlp_server
        self.agent_manager = AgentManager()
        self.api_token = api_token
        self.start_time = time.time()
        self.total_events = 0

        # 订阅 DLPServer 规则下发事件 → 推送给 Agent
        self.dlp_server.event_bus.subscribe(
            EventType.RULE_PUSHED,
            self._on_rule_pushed,
            subscriber_name="api_server",
        )

    def _on_rule_pushed(self, event: Event):
        """当执行层生成规则时，推送给所有相关 Agent"""
        rule_data = {
            "rule_id": event.data.get("rule_id", ""),
            "vql_script": event.data.get("vql_script", ""),
            "target_role": event.data.get("target_role", ""),
        }
        # 在异步事件循环中推送
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(
                    self.agent_manager.push_rule_to_role(
                        rule_data.get("target_role", ""), rule_data
                    )
                )
        except RuntimeError:
            # 无事件循环时存入队列（后续 Agent 通过 GET 拉取）
            pass

    def verify_token(self, token: str) -> bool:
        """验证 API Token"""
        if not self.api_token:
            return True  # 未配置 token 时跳过验证
        return token == self.api_token


def create_app(dlp_server: DLPServer, api_token: str = "") -> FastAPI:
    """创建 FastAPI 应用实例"""

    api_srv = DLPAPIServer(dlp_server, api_token)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("[API] DLP API Server starting...")
        yield
        logger.info("[API] DLP API Server shutting down...")

    app = FastAPI(
        title="DLP Detection System API",
        description="基于用户数据特征画像的终端数据泄露检测系统 — 服务端 API",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 将 api_srv 挂到 app.state 上，方便路由访问
    app.state.api_srv = api_srv

    # -----------------------------------------------------------------
    # 认证依赖
    # -----------------------------------------------------------------

    def _get_api_srv() -> DLPAPIServer:
        return app.state.api_srv

    # -----------------------------------------------------------------
    # 路由：Agent 注册
    # -----------------------------------------------------------------

    @app.post("/api/v1/agents/register", response_model=AgentRegisterResponse)
    async def register_agent(
        req: AgentRegisterRequest,
        srv: DLPAPIServer = Depends(_get_api_srv),
    ):
        """终端代理注册"""
        if not srv.verify_token(req.api_token):
            raise HTTPException(status_code=401, detail="Invalid API token")

        conn = await srv.agent_manager.register(req)

        # 如果对应 user_id 未在 DLPServer 注册，自动创建
        if req.user_id not in srv.dlp_server._users:
            srv.dlp_server.register_user({
                "user_id": req.user_id,
                "role": "default",
                "department": "",
                "projects": [],
                "permissions": [],
                "business_keywords": [],
            })

        logger.info(f"[API] Agent registered: {req.agent_id} (user={req.user_id})")
        return AgentRegisterResponse(
            success=True,
            agent_id=req.agent_id,
            message="registered",
        )

    # -----------------------------------------------------------------
    # 路由：心跳
    # -----------------------------------------------------------------

    @app.post("/api/v1/agents/heartbeat", response_model=HeartbeatResponse)
    async def agent_heartbeat(
        req: HeartbeatRequest,
        srv: DLPAPIServer = Depends(_get_api_srv),
    ):
        """Agent 心跳"""
        conn = await srv.agent_manager.heartbeat(req.agent_id)
        if not conn:
            raise HTTPException(status_code=404, detail="Agent not registered")

        pending = len(conn.pending_rules)
        return HeartbeatResponse(ok=True, pending_rules=pending)

    # -----------------------------------------------------------------
    # 路由：文件事件上报
    # -----------------------------------------------------------------

    @app.post("/api/v1/events", response_model=VerdictResponse)
    async def report_event(
        req: FileEventRequest,
        srv: DLPAPIServer = Depends(_get_api_srv),
    ):
        """终端代理上报文件事件 → 三层检测 → 返回裁决"""
        conn = await srv.agent_manager.get(req.agent_id)
        if not conn:
            raise HTTPException(status_code=404, detail="Agent not registered")

        # 解码文件内容
        try:
            file_content = base64.b64decode(req.file_content_b64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 content")

        # 调用 DLPServer 三层检测
        verdict = srv.dlp_server.process_file_event(
            file_path=req.file_path,
            file_content=file_content,
            user_id=req.user_id,
            action=req.action,
            process_name=req.process_name,
            device_type=req.device_type,
            network_env=req.network_env,
            is_work_hours=req.is_work_hours,
            metadata=req.metadata,
        )

        srv.total_events += 1
        conn.events_processed += 1

        return VerdictResponse(
            response_action=verdict.response_action,
            risk_score=verdict.risk_score,
            sensitivity_level=verdict.sensitivity_level,
            grading_confidence=verdict.grading_confidence,
            user_id=verdict.user_id,
            file_path=verdict.file_path,
            vql_script=verdict.vql_script,
            reason=verdict.reason,
            latency_ms=verdict.latency_ms,
            correlation_id=getattr(verdict, "correlation_id", ""),
        )

    # -----------------------------------------------------------------
    # 路由：规则拉取
    # -----------------------------------------------------------------

    @app.get("/api/v1/rules/{agent_id}", response_model=PendingRulesResponse)
    async def get_pending_rules(
        agent_id: str,
        srv: DLPAPIServer = Depends(_get_api_srv),
    ):
        """Agent 拉取待下发规则（轮询模式）"""
        rules_data = await srv.agent_manager.pop_pending_rules(agent_id)
        rules = [
            RuleItem(
                rule_id=r.get("rule_id", ""),
                vql_script=r.get("vql_script", ""),
                target_role=r.get("target_role", ""),
                sensitivity_level=r.get("sensitivity_level", ""),
                response_action=r.get("response_action", "alert"),
            )
            for r in rules_data
        ]
        return PendingRulesResponse(agent_id=agent_id, rules=rules)

    # -----------------------------------------------------------------
    # 路由：WebSocket 双向通信
    # -----------------------------------------------------------------

    @app.websocket("/api/v1/ws/{agent_id}")
    async def websocket_endpoint(websocket: WebSocket, agent_id: str):
        """WebSocket 长连接 — 规则实时推送 + 事件上报"""
        srv = app.state.api_srv

        conn = await srv.agent_manager.get(agent_id)
        if not conn:
            await websocket.close(code=4004, reason="Agent not registered")
            return

        await websocket.accept()
        await srv.agent_manager.set_websocket(agent_id, websocket)
        logger.info(f"[WS] Agent connected: {agent_id}")

        # 推送积压的规则
        pending = await srv.agent_manager.pop_pending_rules(agent_id)
        for rule in pending:
            await websocket.send_json({"type": "rule_push", "data": rule})

        try:
            while True:
                data = await websocket.receive_json()
                msg_type = data.get("type", "")

                if msg_type == "event":
                    # Agent 通过 WS 上报事件
                    payload = data.get("data", {})
                    file_content = base64.b64decode(payload.get("file_content_b64", ""))
                    verdict = srv.dlp_server.process_file_event(
                        file_path=payload.get("file_path", ""),
                        file_content=file_content,
                        user_id=payload.get("user_id", conn.user_id),
                        action=payload.get("action", "read"),
                        process_name=payload.get("process_name", "explorer.exe"),
                        device_type=payload.get("device_type", "local"),
                        network_env=payload.get("network_env", "internal"),
                        is_work_hours=payload.get("is_work_hours", True),
                    )
                    srv.total_events += 1
                    conn.events_processed += 1

                    await websocket.send_json({
                        "type": "verdict",
                        "data": {
                            "response_action": verdict.response_action,
                            "risk_score": verdict.risk_score,
                            "sensitivity_level": verdict.sensitivity_level,
                            "file_path": verdict.file_path,
                            "vql_script": verdict.vql_script,
                            "reason": verdict.reason,
                            "latency_ms": verdict.latency_ms,
                        },
                    })

                elif msg_type == "heartbeat":
                    await websocket.send_json({"type": "heartbeat_ack", "server_time": time.time()})

        except WebSocketDisconnect:
            logger.info(f"[WS] Agent disconnected: {agent_id}")
        except Exception as e:
            logger.error(f"[WS] Error for {agent_id}: {e}")
        finally:
            await srv.agent_manager.remove_websocket(agent_id)

    # -----------------------------------------------------------------
    # 路由：系统状态
    # -----------------------------------------------------------------

    @app.get("/api/v1/status", response_model=SystemStatusResponse)
    async def get_status(srv: DLPAPIServer = Depends(_get_api_srv)):
        """获取系统状态"""
        stats = srv.dlp_server.get_system_stats()
        return SystemStatusResponse(
            users=stats.get("users", 0),
            initialized=stats.get("initialized", False),
            velociraptor_connected=stats.get("velociraptor_connected", False),
            connected_agents=srv.agent_manager.connected_count,
            total_events_processed=srv.total_events,
            uptime_seconds=time.time() - srv.start_time,
        )

    @app.get("/api/v1/agents")
    async def list_agents(srv: DLPAPIServer = Depends(_get_api_srv)):
        """列出所有已注册 Agent 的状态"""
        return srv.agent_manager.get_all_agents_status()

    return app
