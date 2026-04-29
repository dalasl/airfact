"""
Velociraptor 桥接层

连接 Python DLP 流水线与 Velociraptor（Go）VQL 执行引擎。

架构位置（论文图 4-3）：
    ┌──────────────────────────────────────────────────────────────┐
    │              Python 检测服务端                                │
    │ ┌──────────┐  ┌──────────┐  ┌──────────┐                    │
    │ │ 感知层   │→ │ 认知层   │→ │ 执行层   │                    │
    │ │ ch2画像  │  │ ch3分级  │  │ ch4规则  │                    │
    │ └──────────┘  └──────────┘  └──┬───────┘                    │
    │                                │ VQL 脚本                    │
    │                         ┌──────▼────────┐                   │
    │                         │ VelociraptorBridge (本模块)        │
    │                         │  · gRPC 通道管理                  │
    │                         │  · VQL 提交 & 流式结果接收         │
    │                         │  · Artifact 动态注册              │
    │                         │  · 终端客户端管理                  │
    │                         └──────┬────────┘                   │
    └────────────────────────────────┼────────────────────────────┘
                              gRPC / TLS (50051)
    ┌────────────────────────────────┼────────────────────────────┐
    │          Velociraptor Server (Go)                            │
    │         velociraptor-master/                                 │
    │  ┌─────────────────────────────────────────────┐            │
    │  │  api/api.go        ← gRPC API 服务端         │            │
    │  │  api/query.go      ← Query() 流式 VQL 执行   │            │
    │  │  actions/vql.go    ← 客户端侧 VQL 求值器      │            │
    │  │  accessors/        ← 文件系统/注册表/NTFS等   │            │
    │  │  artifacts/        ← YAML 内置 Artifact 定义  │            │
    │  │  acls/             ← RBAC 权限控制            │            │
    │  └─────────────────────────────────────────────┘            │
    └─────────────────────────────────────────────────────────────┘
                              gRPC / TLS
    ┌─────────────────────────────────────────────────────────────┐
    │          Velociraptor Client (Go agent, 部署于终端)           │
    │  actions/vql.go → VQLClientAction.StartQuery()              │
    │  1. 接收 VQLCollectorArgs（含查询+Artifact）                  │
    │  2. vfilter.Parse() → vql.Eval() → JSONL 编码                │
    │  3. 通过 responder 回传结果到 Server                          │
    └─────────────────────────────────────────────────────────────┘

通信协议（来自 velociraptor-master/api/proto/api.proto）：
    service API {
        rpc Query(VQLCollectorArgs) returns (stream VQLResponse) {}
        rpc CollectArtifact(ArtifactCollectorArgs) returns (ArtifactCollectorResponse) {}
        rpc SetClientMonitoringState(ClientEventTable) returns (Empty) {}
        rpc SetArtifactFile(SetArtifactRequest) returns (SetArtifactResponse) {}
        ...
    }

消息类型（来自 velociraptor-master/actions/proto/vql.proto）：
    VQLCollectorArgs  ← Python 构建，包含 VQL 查询列表 + 环境变量
    VQLResponse       → Go 返回，包含 JSONL 编码的查询结果行
    VQLRequest        ← 单条 VQL 查询（Name + VQL 文本）

依赖：
    pip install pyvelociraptor grpcio protobuf
    （已列入 requirements.txt）
"""

import json
import logging
import os
import time
import yaml
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =====================================================================
# 数据结构
# =====================================================================

@dataclass
class VQLQuery:
    """待提交的 VQL 查询"""
    name: str           # 查询名称（用于标识结果集）
    vql: str            # VQL 查询文本
    env: Dict[str, str] = field(default_factory=dict)  # 环境变量
    timeout: int = 600  # 超时秒数
    max_row: int = 1000 # 每个响应分片的最大行数


@dataclass
class VQLResult:
    """VQL 执行结果"""
    query_name: str
    columns: List[str]
    rows: List[Dict[str, Any]]
    total_rows: int = 0
    log_messages: List[str] = field(default_factory=list)
    error: str = ""
    latency_ms: float = 0.0

    @property
    def success(self) -> bool:
        return not self.error

    def to_dataframe(self):
        """转换为 pandas DataFrame"""
        import pandas as pd
        return pd.DataFrame(self.rows, columns=self.columns)


@dataclass
class ArtifactDefinition:
    """VQL Artifact 定义

    对应 velociraptor-master/artifacts/definitions/ 下的 YAML 格式。
    本系统用于将大模型生成的检测规则封装为 Velociraptor Artifact，
    下发到终端客户端持续执行。
    """
    name: str                    # Artifact 名称（如 Custom.DLP.SensitiveFileMonitor）
    description: str             # 描述
    sources: List[Dict]          # VQL 查询源列表
    parameters: List[Dict] = field(default_factory=list)   # 参数定义
    precondition: str = ""       # 前置条件 VQL
    artifact_type: str = "CLIENT_EVENT"  # CLIENT / CLIENT_EVENT / SERVER / SERVER_EVENT

    def to_yaml(self) -> str:
        """生成 Velociraptor Artifact YAML"""
        artifact = {
            "name": self.name,
            "description": self.description,
            "type": self.artifact_type,
        }
        if self.parameters:
            artifact["parameters"] = self.parameters
        if self.precondition:
            artifact["precondition"] = self.precondition
        artifact["sources"] = self.sources
        return yaml.dump(artifact, allow_unicode=True, default_flow_style=False)


# =====================================================================
# Velociraptor 连接配置
# =====================================================================

@dataclass
class VelociraptorConfig:
    """Velociraptor 服务器连接配置

    pyvelociraptor 通过 API 配置文件（YAML）连接 Velociraptor，
    该文件由 `velociraptor config api_client` 命令生成，
    包含 ca_certificate, client_cert, client_private_key, api_connection_string。
    """
    api_config_path: str = ""       # pyvelociraptor API 配置文件路径
    server_address: str = "localhost:8001"   # gRPC 地址
    use_tls: bool = True
    timeout: int = 600              # 默认查询超时
    max_row: int = 10000            # 默认最大返回行数


# =====================================================================
# 核心桥接类
# =====================================================================

class VelociraptorBridge:
    """Velociraptor gRPC 桥接层

    将 Python DLP 流水线（ch4 生成的 VQL 规则）提交到 Velociraptor Server 执行，
    并接收流式结果。

    三种使用模式：
    1. 即时查询 (query)   — 提交 VQL，等待结果返回
    2. Artifact 注册       — 将 VQL 规则封装为 Artifact，注册到服务器
    3. 客户端事件监控     — 将 Artifact 设置为客户端持续监控事件

    典型使用：
        bridge = VelociraptorBridge(config)
        bridge.connect()

        # 模式1: 即时查询
        result = bridge.execute_vql(
            "SELECT * FROM info()"
        )

        # 模式2: 注册 Artifact + 下发监控
        artifact = bridge.create_dlp_artifact(
            rule_name="SensitiveFileUpload",
            vql_script="SELECT TimeStamp, Filename, FullPath, Reason "
                       "FROM watch_usn(device='C:') WHERE Reason =~ 'FILE_CREATE'",
            user_role="财务人员",
        )
        bridge.register_artifact(artifact)
        bridge.set_client_monitoring(artifact.name, client_ids=["C.xxxx"])

        bridge.close()
    """

    def __init__(self, config: VelociraptorConfig = None):
        self.config = config or VelociraptorConfig()
        self._channel = None
        self._stub = None
        self._connected = False
        self._api_config = None    # pyvelociraptor 原生配置对象

    # -----------------------------------------------------------------
    # 连接管理
    # -----------------------------------------------------------------

    def connect(self) -> bool:
        """建立与 Velociraptor Server 的 gRPC 连接

        底层使用 pyvelociraptor 库，通过 API 配置文件自动处理
        TLS 证书和身份认证。

        Returns:
            是否连接成功
        """
        try:
            import pyvelociraptor
            from pyvelociraptor import api_pb2_grpc

            # 加载 API 配置
            if self.config.api_config_path and os.path.exists(self.config.api_config_path):
                self._api_config = pyvelociraptor.LoadConfigFile(
                    self.config.api_config_path
                )
            else:
                # 尝试默认路径
                default_paths = [
                    os.path.expanduser("~/.velociraptor_api.yaml"),
                    "/etc/velociraptor/api_client.yaml",
                    "configs/velociraptor_api.yaml",
                ]
                for p in default_paths:
                    if os.path.exists(p):
                        self._api_config = pyvelociraptor.LoadConfigFile(p)
                        break

            if self._api_config is None:
                logger.warning(
                    "[VeloBridge] 未找到 Velociraptor API 配置文件，"
                    "使用离线模式（VQL 仅本地校验，不执行）"
                )
                return False

            # 建立 gRPC 通道
            creds = pyvelociraptor.grpc_credentials(self._api_config)
            self._channel = pyvelociraptor.get_channel(self._api_config)
            self._stub = api_pb2_grpc.APIStub(self._channel)
            self._connected = True
            logger.info("[VeloBridge] 已连接 Velociraptor Server")
            return True

        except ImportError:
            logger.warning(
                "[VeloBridge] pyvelociraptor 未安装，使用离线模式。"
                "安装: pip install pyvelociraptor"
            )
            return False
        except Exception as e:
            logger.error(f"[VeloBridge] 连接失败: {e}")
            return False

    def close(self):
        """关闭连接"""
        if self._channel:
            try:
                self._channel.close()
            except Exception:
                pass
        self._connected = False
        logger.info("[VeloBridge] 连接已关闭")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # -----------------------------------------------------------------
    # VQL 查询执行
    # -----------------------------------------------------------------

    def execute_vql(
        self,
        vql: str,
        env: Dict[str, str] = None,
        timeout: int = None,
        max_row: int = None,
    ) -> VQLResult:
        """执行 VQL 查询并返回结果

        对应 Velociraptor gRPC API:
            rpc Query(VQLCollectorArgs) returns (stream VQLResponse) {}

        内部流程：
        1. 构建 VQLCollectorArgs（VQL 文本 + 环境变量 + 限制参数）
        2. 调用 stub.Query() 获取流式响应
        3. 逐个 VQLResponse 解码 JSONL，合并为完整结果集

        Args:
            vql: VQL 查询文本
            env: 环境变量（传递给 VQL scope）
            timeout: 查询超时秒数
            max_row: 最大返回行数

        Returns:
            VQLResult 查询结果
        """
        t_start = time.time()

        if not self._connected:
            # 离线模式：仅校验语法，返回空结果
            return self._offline_execute(vql)

        try:
            from pyvelociraptor import api_pb2

            # 构建请求（对应 actions/proto/vql.proto VQLCollectorArgs）
            request = api_pb2.VQLCollectorArgs(
                max_row=max_row or self.config.max_row,
                timeout=timeout or self.config.timeout,
            )

            # 添加环境变量
            if env:
                for k, v in env.items():
                    request.env.append(
                        api_pb2.VQLEnv(key=k, value=str(v))
                    )

            # 添加 VQL 查询
            request.Query.append(
                api_pb2.VQLRequest(Name="DLPQuery", VQL=vql)
            )

            # 流式执行
            all_rows = []
            columns = []
            logs = []

            for response in self._stub.Query(request):
                # 解析列
                if response.Columns:
                    columns = list(response.Columns)

                # 解析 JSONL 响应
                if response.JSONLResponse:
                    for line in response.JSONLResponse.strip().split("\n"):
                        line = line.strip()
                        if line:
                            try:
                                all_rows.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
                # 旧版 JSON 数组响应
                elif response.Response:
                    try:
                        rows = json.loads(response.Response)
                        if isinstance(rows, list):
                            all_rows.extend(rows)
                    except json.JSONDecodeError:
                        pass

                # 日志
                if response.log:
                    logs.append(response.log)

            latency = (time.time() - t_start) * 1000
            return VQLResult(
                query_name="DLPQuery",
                columns=columns,
                rows=all_rows,
                total_rows=len(all_rows),
                log_messages=logs,
                latency_ms=latency,
            )

        except Exception as e:
            latency = (time.time() - t_start) * 1000
            logger.error(f"[VeloBridge] VQL 执行失败: {e}")
            return VQLResult(
                query_name="DLPQuery",
                columns=[],
                rows=[],
                error=str(e),
                latency_ms=latency,
            )

    def execute_vql_streaming(
        self, vql: str, env: Dict[str, str] = None
    ) -> Iterator[List[Dict]]:
        """流式执行 VQL，逐批返回结果行

        适用于长时间运行的事件监控查询（如 watch_* 插件），
        每个 VQLResponse 分片到达时立即 yield。

        Yields:
            每批结果行 List[Dict]
        """
        if not self._connected:
            return

        try:
            from pyvelociraptor import api_pb2

            request = api_pb2.VQLCollectorArgs(
                max_wait=10,
                max_row=self.config.max_row,
            )
            if env:
                for k, v in env.items():
                    request.env.append(api_pb2.VQLEnv(key=k, value=str(v)))
            request.Query.append(
                api_pb2.VQLRequest(Name="DLPStream", VQL=vql)
            )

            for response in self._stub.Query(request):
                rows = []
                if response.JSONLResponse:
                    for line in response.JSONLResponse.strip().split("\n"):
                        if line.strip():
                            try:
                                rows.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
                if rows:
                    yield rows

        except Exception as e:
            logger.error(f"[VeloBridge] 流式查询失败: {e}")

    # -----------------------------------------------------------------
    # Artifact 管理
    # -----------------------------------------------------------------

    def register_artifact(self, artifact: ArtifactDefinition) -> bool:
        """将 Artifact 注册到 Velociraptor Server

        对应 gRPC API:
            rpc SetArtifactFile(SetArtifactRequest) returns (SetArtifactResponse) {}

        大模型生成的 VQL 规则通过此方法封装为 Artifact 并注册，
        之后可通过 set_client_monitoring() 下发到终端持续执行。

        Args:
            artifact: Artifact 定义

        Returns:
            是否注册成功
        """
        artifact_yaml = artifact.to_yaml()

        if not self._connected:
            logger.info(
                f"[VeloBridge] 离线模式，Artifact 已生成但未注册:\n"
                f"  名称: {artifact.name}\n"
                f"  类型: {artifact.artifact_type}\n"
                f"  YAML 长度: {len(artifact_yaml)} 字符"
            )
            return False

        try:
            # 通过 VQL 注册 Artifact（等效于 Web UI 的 Artifact 编辑器）
            register_vql = (
                f"SELECT artifact_set(definition='''{artifact_yaml}''') "
                f"FROM scope()"
            )
            result = self.execute_vql(register_vql)
            if result.success:
                logger.info(f"[VeloBridge] Artifact 已注册: {artifact.name}")
                return True
            else:
                logger.error(f"[VeloBridge] Artifact 注册失败: {result.error}")
                return False
        except Exception as e:
            logger.error(f"[VeloBridge] Artifact 注册异常: {e}")
            return False

    def set_client_monitoring(
        self,
        artifact_name: str,
        client_ids: List[str] = None,
        parameters: Dict[str, str] = None,
    ) -> bool:
        """将 Artifact 设置为客户端事件监控

        对应 gRPC API:
            rpc SetClientMonitoringState(ClientEventTable) returns (Empty) {}

        让指定的终端客户端持续执行该 Artifact 中的 VQL 查询，
        实现实时检测。

        Args:
            artifact_name: 已注册的 Artifact 名称
            client_ids: 目标客户端 ID 列表（None=所有客户端）
            parameters: Artifact 运行参数

        Returns:
            是否设置成功
        """
        if not self._connected:
            logger.info(f"[VeloBridge] 离线模式，跳过客户端监控设置: {artifact_name}")
            return False

        try:
            # 通过服务端 VQL 修改客户端监控事件表
            params_json = json.dumps(parameters or {})
            set_vql = (
                f"SELECT add_client_monitoring("
                f"artifact='{artifact_name}', "
                f"parameters={params_json}) "
                f"FROM scope()"
            )
            result = self.execute_vql(set_vql)
            if result.success:
                logger.info(
                    f"[VeloBridge] 客户端监控已设置: {artifact_name} "
                    f"→ {len(client_ids) if client_ids else '所有'} 个终端"
                )
                return True
            else:
                logger.error(f"[VeloBridge] 监控设置失败: {result.error}")
                return False
        except Exception as e:
            logger.error(f"[VeloBridge] 监控设置异常: {e}")
            return False

    def subscribe_client_events(
        self,
        artifact_prefix: str = "Custom.DLP.",
        callback: Callable = None,
        stop_event: "threading.Event" = None,
    ) -> None:
        """持续监听终端客户端回传的检测事件（上行通道）

        通过服务端 VQL watch_monitoring() 订阅指定 Artifact 前缀的
        客户端事件流，收到事件后调用 callback 送入三层流水线处理。

        这是论文架构中 VM → Server 事件闭环的实现：
            终端 VQL 执行 (watch_usn/watch_etw/diff)
            → 匹配结果回传 Velociraptor Server
            → 本方法流式接收
            → callback(event) → DLPServer.process_file_event()
            → 画像增量更新 + 敏感分级 + 规则生成
            → 新规则通过 register_artifact() 下发

        Args:
            artifact_prefix: 监听的 Artifact 名称前缀
            callback: 事件处理回调，签名 callback(event_dict) -> None
            stop_event: 外部停止信号（threading.Event）
        """
        if not self._connected:
            logger.info("[VeloBridge] 离线模式，事件订阅未启动")
            return

        # 服务端 VQL：监听所有 DLP Artifact 的客户端事件
        subscribe_vql = (
            f"SELECT * FROM watch_monitoring("
            f"artifact='{artifact_prefix}')"
        )

        logger.info(
            f"[VeloBridge] 开始订阅终端事件 (artifact={artifact_prefix})"
        )

        try:
            for batch in self.execute_vql_streaming(subscribe_vql):
                if stop_event and stop_event.is_set():
                    logger.info("[VeloBridge] 事件订阅收到停止信号")
                    break

                for event in batch:
                    client_id = event.get("ClientId", "unknown")
                    artifact = event.get("Artifact", "")
                    logger.debug(
                        f"[VeloBridge] 收到终端事件: "
                        f"client={client_id} artifact={artifact}"
                    )
                    if callback:
                        try:
                            callback(event)
                        except Exception as e:
                            logger.error(
                                f"[VeloBridge] 事件回调处理失败: {e}"
                            )

        except Exception as e:
            if stop_event and stop_event.is_set():
                return
            logger.error(f"[VeloBridge] 事件订阅异常: {e}")

    def collect_artifact(
        self,
        client_id: str,
        artifact_name: str,
        parameters: Dict[str, str] = None,
        timeout: int = 600,
    ) -> Optional[str]:
        """在指定客户端上采集 Artifact（一次性执行）

        对应 gRPC API:
            rpc CollectArtifact(ArtifactCollectorArgs) returns (ArtifactCollectorResponse) {}

        Returns:
            flow_id（用于后续查询结果），失败返回 None
        """
        if not self._connected:
            return None

        try:
            params_json = json.dumps(parameters or {})
            collect_vql = (
                f"SELECT collect_client("
                f"client_id='{client_id}', "
                f"artifacts='{artifact_name}', "
                f"env=dict({', '.join(f'{k}={repr(v)}' for k, v in (parameters or {}).items())}), "
                f"timeout={timeout}) "
                f"FROM scope()"
            )
            result = self.execute_vql(collect_vql)
            if result.success and result.rows:
                flow_id = result.rows[0].get("flow_id", "")
                logger.info(f"[VeloBridge] Artifact 采集已启动: {flow_id}")
                return flow_id
            return None
        except Exception as e:
            logger.error(f"[VeloBridge] Artifact 采集失败: {e}")
            return None

    # -----------------------------------------------------------------
    # 终端客户端管理
    # -----------------------------------------------------------------

    def list_clients(self, search: str = "", limit: int = 100) -> List[Dict]:
        """列出已注册的 Velociraptor 客户端"""
        vql = f"SELECT * FROM clients(search='{search}') LIMIT {limit}"
        result = self.execute_vql(vql)
        return result.rows if result.success else []

    def get_client_info(self, client_id: str) -> Optional[Dict]:
        """获取客户端详细信息"""
        vql = f"SELECT * FROM clients(client_id='{client_id}')"
        result = self.execute_vql(vql)
        return result.rows[0] if result.success and result.rows else None

    def label_clients(self, client_ids: List[str], labels: List[str]) -> bool:
        """给客户端打标签（用于按角色分组）

        论文场景：将三台终端（VM-1/2/3）按角色打标签，
        便于按角色下发不同的检测 Artifact。
        """
        for cid in client_ids:
            for label in labels:
                vql = f"SELECT label(client_id='{cid}', labels=['{label}'], op='set') FROM scope()"
                self.execute_vql(vql)
        return True

    # -----------------------------------------------------------------
    # DLP 专用 Artifact 工厂
    # -----------------------------------------------------------------

    def create_dlp_artifact(
        self,
        rule_name: str,
        vql_script: str,
        user_role: str = "",
        sensitivity_level: str = "L3",
        response_action: str = "alert",
        description: str = "",
    ) -> ArtifactDefinition:
        """创建 DLP 检测 Artifact

        将 ch4 生成的完整 VQL 脚本封装为 Velociraptor Artifact。

        支持三种插件类型的 VQL（由 VQLCompiler 生成）：
        - watch_usn: 文件操作监控（read/write/copy/download）
        - watch_etw: 网络/邮件/打印事件监控（upload/send/print）
        - diff + glob: USB 外设文件变化检测

        Args:
            rule_name: 规则名称（将转为 Custom.DLP.{rule_name}）
            vql_script: 完整的 VQL 脚本（来自 ch4 VQLCompiler 输出）
            user_role: 目标用户角色（用于 Artifact 描述）
            sensitivity_level: 目标敏感等级
            response_action: 响应动作 (alert/block)
            description: 规则描述

        Returns:
            ArtifactDefinition 可注册的 Artifact 定义
        """
        artifact_name = f"Custom.DLP.{rule_name}"

        if not description:
            description = (
                f"DLP 检测规则 - {rule_name}\n"
                f"目标角色: {user_role or '全部'}\n"
                f"敏感等级: {sensitivity_level}\n"
                f"响应动作: {response_action}\n"
                f"由大模型动态生成"
            )

        sources = [{
            "name": rule_name,
            "query": vql_script,
        }]

        parameters = [
            {"name": "SensitivityLevel", "default": sensitivity_level,
             "description": "最低敏感等级阈值"},
            {"name": "ResponseAction", "default": response_action,
             "description": "响应动作 (alert/block)"},
            {"name": "UserRole", "default": user_role,
             "description": "目标用户角色"},
        ]

        return ArtifactDefinition(
            name=artifact_name,
            description=description,
            sources=sources,
            parameters=parameters,
            artifact_type="CLIENT_EVENT",
        )

    def create_file_access_artifact(
        self,
        watch_path: str = "C:/Users/**",
        extensions: List[str] = None,
        vql_filter: str = "",
    ) -> ArtifactDefinition:
        """创建文件访问监控 Artifact

        利用 Velociraptor 的 watch_etw / watch_usn 等原生能力
        监控指定路径下的文件访问事件。

        Args:
            watch_path: 监控路径（glob 模式）
            extensions: 监控的文件扩展名列表
            vql_filter: 额外的 VQL 过滤条件
        """
        ext_list = extensions or [".docx", ".xlsx", ".pdf", ".pptx", ".csv"]
        ext_regex = "|".join(e.replace(".", r"\.") for e in ext_list)

        vql = (
            "SELECT Timestamp, FullPath, FileName,\n"
            "       Reason, Process.Name AS ProcessName,\n"
            "       Process.Pid AS ProcessPid\n"
            "FROM watch_usn(\n"
            f"    device='C:',\n"
            f"    path_regex='{watch_path.replace('/', chr(92) + chr(92))}')\n"
            f"WHERE FileName =~ '({ext_regex})$'\n"
        )
        if vql_filter:
            vql += f"  AND {vql_filter}\n"

        return ArtifactDefinition(
            name="Custom.DLP.FileAccessMonitor",
            description="终端文件访问实时监控（USN Journal）",
            sources=[{"name": "FileAccess", "query": vql}],
            artifact_type="CLIENT_EVENT",
        )

    # -----------------------------------------------------------------
    # 内部辅助
    # -----------------------------------------------------------------

    def _offline_execute(self, vql: str) -> VQLResult:
        """离线模式：仅校验 VQL 语法，返回空结果"""
        from src.ch4_rule_generation.vql_compiler import VQLCompiler

        syntax_ok = VQLCompiler.check_syntax(vql)
        if not syntax_ok:
            return VQLResult(
                query_name="offline",
                columns=[],
                rows=[],
                error="VQL 语法校验失败",
            )

        return VQLResult(
            query_name="offline",
            columns=[],
            rows=[],
            log_messages=["离线模式：VQL 语法校验通过，未实际执行"],
        )

    def health_check(self) -> Dict[str, Any]:
        """检查 Velociraptor 服务健康状态"""
        if not self._connected:
            return {"status": "disconnected", "mode": "offline"}

        try:
            result = self.execute_vql(
                "SELECT server_version(), server_uptime() FROM scope()"
            )
            if result.success and result.rows:
                return {
                    "status": "healthy",
                    "mode": "online",
                    "server_info": result.rows[0],
                    "latency_ms": result.latency_ms,
                }
            return {"status": "unhealthy", "mode": "online", "error": result.error}
        except Exception as e:
            return {"status": "error", "mode": "online", "error": str(e)}


# =====================================================================
# 流水线集成辅助函数
# =====================================================================

def integrate_with_pipeline(pipeline, bridge: VelociraptorBridge):
    """将 VelociraptorBridge 集成到 DLPPipeline

    在 pipeline 检测出高风险事件后，自动将生成的 VQL 规则
    通过 Velociraptor 下发到终端执行。

    Args:
        pipeline: DLPPipeline 实例
        bridge: VelociraptorBridge 实例
    """
    original_process = pipeline.process_file_event

    def enhanced_process(*args, **kwargs):
        verdict = original_process(*args, **kwargs)

        # VQL 规则非空 + 非静默放行 → 下发到 Velociraptor
        if verdict.vql_script and verdict.response_action != "silent_monitoring":
            try:
                artifact = bridge.create_dlp_artifact(
                    rule_name=f"AutoRule_{int(time.time())}",
                    vql_script=verdict.vql_script,
                    user_role=verdict.user_id,
                    sensitivity_level=verdict.sensitivity_level,
                    response_action=verdict.response_action,
                )
                bridge.register_artifact(artifact)
                logger.info(
                    f"[集成] VQL 规则已下发到 Velociraptor: "
                    f"{artifact.name} ({verdict.response_action})"
                )
            except Exception as e:
                logger.warning(f"[集成] VQL 下发失败: {e}")

        return verdict

    pipeline.process_file_event = enhanced_process
    logger.info("[集成] DLPPipeline ↔ Velociraptor 桥接已建立")
