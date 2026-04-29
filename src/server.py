"""
终端数据泄露检测系统 — 服务端部署入口

将论文三层架构实现为三个独立服务模块，通过事件总线进行跨模块数据传递：

    ┌────────────────────────────────────────────────────────────────┐
    │                         DLPServer                              │
    │                                                                │
    │  ┌─────────────────┐   ProfileMatchResult   ┌────────────────┐│
    │  │ PerceptionService│ ─────────────────────► │CognitionService││
    │  │ (感知层 ch2)     │                        │(认知层 ch3)    ││
    │  │ · 文档解析       │   PROFILE_DRIFT ───┐   │· 敏感分级      ││
    │  │ · 画像匹配       │                    │   │· 资产目录      ││
    │  │ · 聚类 & 漂移    │   ◄── CATALOG_UPD ─┘   │· 自洽性校验    ││
    │  └────────┬────────┘                        └───────┬────────┘│
    │           │                                         │         │
    │           │ FILE_EVENT                   GradingResult         │
    │           │                                         │         │
    │  ┌────────▼────────────────────────────────────────▼────────┐ │
    │  │                  ExecutionService                         │ │
    │  │                  (执行层 ch4)                              │ │
    │  │ · 规则生成 (IR → VQL)                                     │ │
    │  │ · 风险决策 (三维偏离评分)                                   │ │
    │  │ · 动态白名单                                               │ │
    │  └────────┬───────────────────────────────────────┬─────────┘ │
    │           │ DetectionVerdict                      │ VQL 规则   │
    │           ▼                                      ▼            │
    │      EventBus                          VelociraptorBridge     │
    │   (事件总线广播)                       (gRPC → Go 执行引擎)    │
    └────────────────────────────────────────────────────────────────┘
                        │                              │
                   gRPC/Event                     gRPC/TLS
                        ▼                              ▼
                 TerminalAgent                  Velociraptor Server
                 (终端代理)                     (velociraptor-master/)

每个 Service 具有独立生命周期、配置、状态，通过事件总线解耦。
模块间传递的数据对象：
    · ProfileMatchResult  — 感知层 → 认知层 / 执行层
    · GradingResult       — 认知层 → 执行层
    · DetectionVerdict    — 执行层 → 终端代理 / 日志 / 告警
    · RuleDeployment      — 执行层 → Velociraptor / 终端代理
"""

import hashlib
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ---- 事件总线 ----
from src.event_bus import EventBus, Event, EventType, create_default_bus

# ---- 感知层（第2章） ----
from src.ch2_user_profiling.unet_denoiser import UNetDenoiser
from src.ch2_user_profiling.document_parser import DocumentParser, LRUDocumentCache
from src.ch2_user_profiling.temporal_alignment import (
    TemporalAligner, RawEvent, EventType as EvtType, StructuredEvent,
)
from src.ch2_user_profiling.feature_extraction import FeatureExtractor
from src.ch2_user_profiling.fusion_encoder import DualBranchFusionEncoder
from src.ch2_user_profiling.igw_kmeans import IGWKMeans, ClusterResult
from src.ch2_user_profiling.incremental_update import (
    IncrementalUpdater, ProfileVersion, UpdateTriggerType,
)

# ---- 认知层（第3章） ----
from src.ch3_sensitive_grading.prompt_builder import PromptBuilder, UserProfile
from src.ch3_sensitive_grading.rag_retriever import RAGRetriever
from src.ch3_sensitive_grading.self_consistency import (
    SelfConsistencyVerifier, GradingDecision,
)
from src.ch3_sensitive_grading.asset_catalog import SensitiveAssetCatalog
from src.ch3_sensitive_grading.grading_pipeline import GradingPipeline

# ---- 执行层（第4章） ----
from src.ch4_rule_generation.quadruple_ir import QuadrupleIR
from src.ch4_rule_generation.template_synthesizer import TemplateSynthesizer
from src.ch4_rule_generation.vql_compiler import VQLCompiler, VQLScript
from src.ch4_rule_generation.adaptive_decision import (
    AdaptiveDecisionEngine, OperationEvent, DecisionResult,
    ProfileBaseline, ResponseAction,
)

# ---- Velociraptor 桥接（可选） ----
try:
    from src.velociraptor_bridge import (
        VelociraptorBridge, VelociraptorConfig, ArtifactDefinition,
    )
    _HAS_VELO_BRIDGE = True
except ImportError:
    _HAS_VELO_BRIDGE = False


# =====================================================================
# 模块间数据传递对象
# =====================================================================

@dataclass
class ProfileMatchResult:
    """感知层 → 认知层 / 执行层 的数据传递

    感知层完成文档解析和用户画像匹配后，将结果封装为此对象，
    同时发送给认知层（用于分级上下文）和执行层（用于行为偏离评估）。
    """
    user_id: str
    user_profile: UserProfile        # 用户画像（传给 ch3 提示工程）
    cluster_id: int                  # 所属聚类编号
    profile_version: int             # 画像版本号
    feature_vector: np.ndarray       # 融合后特征向量 (896d)
    behavior_pattern: np.ndarray     # 行为特征 (96d)
    content_text: str                # 文档提取文本
    drift_detected: bool = False     # 是否检测到画像漂移
    correlation_id: str = ""


@dataclass
class GradingResult:
    """认知层 → 执行层 的数据传递

    认知层完成敏感数据分级后，将分级结果封装为此对象，
    传递给执行层进行风险决策和规则生成。
    """
    file_path: str
    level: str                      # L1 / L2 / L3 / L4
    effective_level: str            # 置信度升格后的等级
    confidence: float               # 分级置信度
    is_accepted: bool               # 是否通过自洽性校验
    needs_human_review: bool        # 是否需人工复核
    category: str = ""              # 资产类别
    correlation_id: str = ""


@dataclass
class RuleDeployment:
    """执行层 → Velociraptor / 终端代理 的数据传递

    执行层生成 VQL 规则后，将规则和部署信息封装为此对象，
    下发到 Velociraptor Server 或直接推送到终端代理。
    """
    rule_id: str
    vql_script: str
    target_role: str                # 目标用户角色
    sensitivity_level: str          # 针对的敏感等级
    response_action: str            # alert / block
    artifact_yaml: str = ""         # Velociraptor Artifact YAML
    deployed: bool = False
    correlation_id: str = ""


@dataclass
class DetectionVerdict:
    """端到端检测最终裁决 — 各模块数据的聚合输出"""
    # --- 执行层决策 ---
    response_action: str
    risk_score: float

    # --- 认知层结果 ---
    sensitivity_level: str
    grading_confidence: float
    grading_accepted: bool
    needs_human_review: bool

    # --- 感知层上下文 ---
    user_id: str
    user_cluster: int
    profile_version: int

    # --- 风险评分分解 ---
    subject_consistency: float
    behavior_deviation: float
    env_confidence: float

    # --- 规则信息 ---
    vql_script: str = ""
    rule_template: str = ""

    # --- 元数据 ---
    file_path: str = ""
    action: str = ""
    timestamp: float = field(default_factory=time.time)
    latency_ms: float = 0.0
    reason: str = ""

    def is_blocked(self) -> bool:
        return self.response_action == "block"

    def summary(self) -> str:
        emoji = {"block": "X", "alert": "!", "silent_monitoring": "OK"}
        e = emoji.get(self.response_action, "?")
        return (
            f"[{e}] [{self.response_action}] "
            f"file={self.file_path}  user={self.user_id}  "
            f"level={self.sensitivity_level}  risk={self.risk_score:.3f}  "
            f"latency={self.latency_ms:.0f}ms"
        )


# =====================================================================
# 用户注册表
# =====================================================================

@dataclass
class UserRecord:
    """已注册用户的完整画像记录"""
    user_id: str
    role: str
    department: str
    projects: List[str]
    permissions: List[str]
    business_keywords: List[str]
    cluster_id: int = -1
    profile_version: int = 0
    feature_vector: Optional[np.ndarray] = None
    content_semantic: Optional[np.ndarray] = None
    behavior_pattern: Optional[np.ndarray] = None
    permission_context: Optional[np.ndarray] = None

    def to_user_profile(self) -> UserProfile:
        return UserProfile(
            user_id=self.user_id,
            role=self.role,
            department=self.department,
            projects=self.projects,
            permissions=self.permissions,
            business_keywords=self.business_keywords,
            cluster_id=self.cluster_id,
        )

    def to_profile_vector(self) -> np.ndarray:
        if self.feature_vector is not None:
            return self.feature_vector
        parts = []
        for arr in [self.content_semantic, self.behavior_pattern, self.permission_context]:
            if arr is not None:
                parts.append(arr)
        if parts:
            return np.concatenate(parts)
        return np.zeros(896)


# =====================================================================
# 感知层服务
# =====================================================================

class PerceptionService:
    """感知层服务（第2章）

    职责：
    1. 接收文件事件，解析文档内容（U-Net去噪 → OCR → LayoutLMv3 → SBERT）
    2. 构建三维用户特征（内容语义768d + 行为模式96d + 权限上下文32d）
    3. 交叉注意力融合编码
    4. 检测画像漂移，触发增量更新
    5. 输出 ProfileMatchResult → 发送给认知层和执行层

    订阅事件：FILE_ACCESS
    发布事件：PROFILE_MATCHED, PROFILE_DRIFT, PROFILE_UPDATED
    """

    # 支持图像解析的文件扩展名
    _IMAGE_EXTS = {"tif", "tiff", "png", "jpg", "jpeg", "bmp"}
    # 需要渲染为图像再解析的文件扩展名
    _RENDER_EXTS = {"pdf"}
    # 纯文本类文件扩展名
    _TEXT_EXTS = {"txt", "csv", "log", "json", "xml", "md", "docx", "xlsx"}

    # 默认权限编码词表（覆盖论文实验的 VM 角色体系）
    _DEFAULT_ROLE_VOCAB = [
        "office_staff", "finance", "ops_admin", "core_business",
        "hr", "legal", "rd_engineer", "manager",
    ]
    _DEFAULT_DEPT_VOCAB = [
        "行政", "财务", "运维", "安全", "技术", "产品", "法务", "人力",
    ]
    _DEFAULT_PROJECT_VOCAB = [
        "Phoenix", "Atlas", "Titan", "Nova", "Omega", "Apex", "Nebula", "default",
    ]
    _DEFAULT_PERM_VOCAB = [
        "read", "write", "execute", "admin", "export", "print", "share", "delete",
    ]

    def __init__(self, config: Dict, device: str = "cpu"):
        prof_cfg = config.get("profiling", {})
        self.device = device

        # 文档去噪
        self.denoiser = UNetDenoiser(
            alpha=prof_cfg.get("unet", {}).get("loss", {}).get("alpha", 0.7),
        )
        unet_weights = config.get("models", {}).get("unet_weights", "")
        if unet_weights and os.path.isfile(unet_weights):
            import torch as _torch
            self.denoiser.load_state_dict(
                _torch.load(unet_weights, map_location=device, weights_only=True)
            )
        self.denoiser = self.denoiser.to(device).eval() if device != "cpu" else self.denoiser.eval()

        # 文档解析器（LayoutLMv3 + SBERT + OCR + 缓存）
        self.doc_parser = DocumentParser(
            cache_capacity=prof_cfg.get("cache", {}).get("capacity", 500),
            top_k_keywords=prof_cfg.get("features", {}).get("tfidf_top_k", 20),
            device=device,
        )

        # 时空对齐
        self.temporal_aligner = TemporalAligner(
            doc_cache=self.doc_parser.cache,
            delta_seconds=prof_cfg.get("temporal_alignment", {}).get("delta_seconds", 5.0),
        )

        # 三维特征提取
        self.feature_extractor = FeatureExtractor()

        # 融合编码器
        fusion_cfg = prof_cfg.get("fusion_encoder", {})
        self.fusion_encoder = DualBranchFusionEncoder(
            left_dim=fusion_cfg.get("left_dim", 864),
            right_dim=fusion_cfg.get("right_dim", 32),
            hidden_dim=fusion_cfg.get("hidden_dim", 256),
            num_heads=fusion_cfg.get("num_heads", 8),
            dropout=fusion_cfg.get("dropout", 0.1),
        )
        self.fusion_encoder.eval()

        # IGW-Kmeans 聚类
        self.igw_kmeans = IGWKMeans(
            K=prof_cfg.get("igw_kmeans", {}).get("K", 3),
            T_max=prof_cfg.get("igw_kmeans", {}).get("T_max", 100),
            epsilon=prof_cfg.get("igw_kmeans", {}).get("epsilon", 1e-4),
        )

        # 增量更新
        self.incremental_updater = IncrementalUpdater(
            eta=prof_cfg.get("incremental_update", {}).get("eta", 0.15),
            theta_drift=prof_cfg.get("incremental_update", {}).get("theta_drift", 2.5),
        )

        self.cluster_result: Optional[ClusterResult] = None
        self.profile_version: Optional[ProfileVersion] = None

    def init_clusters(self, features: np.ndarray) -> ClusterResult:
        """初始聚类（系统冷启动时调用）"""
        self.cluster_result = self.igw_kmeans.fit(features)
        return self.cluster_result

    def process(
        self,
        file_content: bytes,
        file_path: str,
        user: UserRecord,
        action: str,
        process_name: str,
        device_type: str,
        network_env: str,
        is_work_hours: bool,
        correlation_id: str = "",
    ) -> ProfileMatchResult:
        """感知层处理：文档解析 + 特征构建 + 画像匹配 + 漂移检测

        完整流程:
        1. 文档语义提取 (U-Net → OCR → LayoutLMv3 → SBERT 或 纯文本 → SBERT)
        2. 行为特征构建 (48时间槽 + 32操作序列 + 16资源访问 = 96d)
        3. 权限上下文编码 (角色/部门/项目 multi-hot = 32d)
        4. 三维特征拼接 (768 + 96 + 32 = 896d)
        5. 漂移检测 (马氏距离)
        6. 输出 ProfileMatchResult

        Returns:
            ProfileMatchResult 传递给下游模块
        """
        import datetime

        # ----------------------------------------------------------
        # 1. 文档内容语义提取
        # ----------------------------------------------------------
        content_text = self._extract_text(file_content, file_path)

        ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
        content_semantic = self._compute_content_semantic(
            file_content, ext, content_text, user,
        )

        # ----------------------------------------------------------
        # 2. 行为特征构建 (96d)
        # ----------------------------------------------------------
        dt = datetime.datetime.now()
        timestamp = dt.timestamp()
        behavior_pattern = self._compute_behavior_pattern(
            action, timestamp, device_type, network_env, is_work_hours,
        )

        # ----------------------------------------------------------
        # 3. 权限上下文编码 (32d)
        # ----------------------------------------------------------
        permission_context = self.feature_extractor.compute_permission_context(
            role=user.role,
            department=user.department,
            projects=user.projects,
            permissions=user.permissions,
            role_vocab=self._DEFAULT_ROLE_VOCAB,
            dept_vocab=self._DEFAULT_DEPT_VOCAB,
            project_vocab=self._DEFAULT_PROJECT_VOCAB,
            perm_vocab=self._DEFAULT_PERM_VOCAB,
        )

        # ----------------------------------------------------------
        # 4. 更新用户记录中的特征
        # ----------------------------------------------------------
        user.content_semantic = content_semantic
        user.behavior_pattern = behavior_pattern
        user.permission_context = permission_context
        feature_vector = user.to_profile_vector()  # 896d 拼接

        # ----------------------------------------------------------
        # 5. 画像漂移检测
        # ----------------------------------------------------------
        drift_detected = False
        self.incremental_updater.record_operation(user.user_id, feature_vector)

        if self.cluster_result is not None and user.cluster_id >= 0:
            centers = self.cluster_result.centers
            cov = np.eye(feature_vector.shape[0]) * 0.1
            if (self.profile_version is not None
                    and self.profile_version.covariance_matrices is not None
                    and user.cluster_id < len(self.profile_version.covariance_matrices)):
                cov = self.profile_version.covariance_matrices[user.cluster_id]

            drift_result = self.incremental_updater.detect_drift(
                user_id=user.user_id,
                cluster_center=centers[user.cluster_id],
                covariance_matrix=cov,
            )
            drift_detected = drift_result.is_drifted

        return ProfileMatchResult(
            user_id=user.user_id,
            user_profile=user.to_user_profile(),
            cluster_id=user.cluster_id,
            profile_version=user.profile_version,
            feature_vector=feature_vector,
            behavior_pattern=behavior_pattern,
            content_text=content_text,
            drift_detected=drift_detected,
            correlation_id=correlation_id,
        )

    def _compute_content_semantic(
        self,
        file_content: bytes,
        ext: str,
        content_text: str,
        user: UserRecord,
    ) -> np.ndarray:
        """计算内容语义向量 (768d)

        图像类文档: U-Net去噪 → OCR → LayoutLMv3 → SBERT
        文本类文档: 提取文本 → SBERT
        """
        # 图像类文档走多模态路径
        if ext in self._IMAGE_EXTS:
            try:
                from PIL import Image
                import io
                import torch

                img = Image.open(io.BytesIO(file_content)).convert("RGB")
                img_array = np.array(img)

                # U-Net 去噪
                tensor = torch.from_numpy(img_array).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                tensor = tensor.to(self.device) if self.device != "cpu" else tensor
                with torch.no_grad():
                    denoised = self.denoiser(tensor)
                denoised_array = (denoised.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

                # DocumentParser: OCR + LayoutLMv3 + SBERT
                doc_repr = self.doc_parser.parse_document(denoised_array, file_content)
                return doc_repr.semantic_vector
            except Exception:
                pass

        # PDF: 渲染首页为图像后走多模态路径
        if ext in self._RENDER_EXTS:
            try:
                import fitz
                pdf_doc = fitz.open(stream=file_content, filetype="pdf")
                if len(pdf_doc) > 0:
                    page = pdf_doc[0]
                    pix = page.get_pixmap(dpi=150)
                    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                        pix.height, pix.width, pix.n
                    )
                    if pix.n == 4:
                        img_array = img_array[:, :, :3]
                    doc_repr = self.doc_parser.parse_document(img_array, file_content)
                    return doc_repr.semantic_vector
            except Exception:
                pass

        # 文本类文档: 直接 SBERT 编码
        if content_text:
            self.doc_parser._load_sbert()
            return self.doc_parser._sbert_encode(content_text)

        # 回退: 零向量
        return user.content_semantic if user.content_semantic is not None else np.zeros(768)

    def _compute_behavior_pattern(
        self,
        action: str,
        timestamp: float,
        device_type: str,
        network_env: str,
        is_work_hours: bool,
    ) -> np.ndarray:
        """计算行为模式特征 (96d)

        维度分配:
        - [0:48]  时间分布 (30min粒度, 归一化概率)
        - [48:80] 操作序列 (trigram条件概率, 需累积)
        - [80:96] 资源访问 (设备/网络/时段标记 + 预留)
        """
        import datetime

        behavior = np.zeros(96)

        # 时间分布 (48-bin, 30min粒度)
        dt = datetime.datetime.fromtimestamp(timestamp)
        time_bin = (dt.hour * 60 + dt.minute) // 30
        if 0 <= time_bin < 48:
            behavior[time_bin] = 1.0

        # 操作类型编码 (48:56, 8种操作)
        action_idx = {
            "read": 0, "write": 1, "copy": 2, "upload": 3,
            "send": 4, "print": 5, "download": 6, "delete": 7,
        }
        aidx = action_idx.get(action.lower(), 0)
        behavior[48 + aidx] = 1.0

        # 资源/环境编码 (80:96)
        if device_type == "usb":
            behavior[80] = 1.0
        if network_env == "external":
            behavior[81] = 1.0
        if not is_work_hours:
            behavior[82] = 1.0
        # 设备类型细分
        device_map = {"local": 0, "usb": 1, "network": 2, "cloud": 3}
        didx = device_map.get(device_type, 0)
        behavior[84 + didx] = 1.0

        return behavior

    @staticmethod
    def _extract_text(content: bytes, file_path: str) -> str:
        ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
        try:
            if ext in ("txt", "csv", "log", "json", "xml", "md"):
                return content.decode("utf-8", errors="ignore")[:5000]
            elif ext == "docx":
                from docx import Document
                from io import BytesIO
                doc = Document(BytesIO(content))
                return " ".join(p.text for p in doc.paragraphs)[:5000]
            elif ext == "pdf":
                import fitz
                doc = fitz.open(stream=content, filetype="pdf")
                return " ".join(page.get_text() for page in doc)[:5000]
            elif ext == "xlsx":
                from openpyxl import load_workbook
                from io import BytesIO
                wb = load_workbook(BytesIO(content), read_only=True)
                texts = []
                for ws in wb.worksheets:
                    for row in ws.iter_rows(values_only=True):
                        texts.extend(str(c) for c in row if c is not None)
                return " ".join(texts)[:5000]
        except Exception:
            pass
        return content.decode("utf-8", errors="ignore")[:2000]


# =====================================================================
# 认知层服务
# =====================================================================

class CognitionService:
    """认知层服务（第3章）

    职责：
    1. 接收 ProfileMatchResult，结合用户画像上下文构建提示
    2. RAG 检索相似案例，增强分级提示
    3. 多轮自洽性校验，确保分级可靠
    4. 输出 GradingResult → 发送给执行层
    5. 维护敏感资产目录，等级变更时通知执行层更新规则

    订阅事件：PROFILE_MATCHED
    发布事件：GRADING_DONE, GRADING_ESCALATED, GRADING_REVIEW, CATALOG_UPDATED
    """

    def __init__(self, config: Dict, llm_fn: Callable = None, encode_fn: Callable = None):
        grad_cfg = config.get("grading", {})

        # 如果未提供 llm_fn，尝试加载 LLM（按优先级）
        if llm_fn is None:
            qwen_cfg = config.get("models", {}).get("qwen", {})
            if qwen_cfg.get("enabled", False):
                mode = qwen_cfg.get("mode", "local")
                if mode == "api":
                    # API 模式：DashScope / OpenAI 兼容 API
                    from src.models.qwen_api_wrapper import QwenAPIWrapper
                    self._qwen = QwenAPIWrapper(
                        api_key=qwen_cfg.get("api_key", ""),
                        model=qwen_cfg.get("api_model", "qwen-plus"),
                        backend=qwen_cfg.get("api_backend", "dashscope"),
                        base_url=qwen_cfg.get("api_base_url", ""),
                        max_tokens=qwen_cfg.get("max_new_tokens", 512),
                        temperature=qwen_cfg.get("temperature", 0.7),
                        top_p=qwen_cfg.get("top_p", 0.9),
                    )
                    llm_fn = self._qwen
                else:
                    # 本地 GPU 模式：4-bit 量化推理
                    from src.models.qwen_wrapper import QwenWrapper
                    self._qwen = QwenWrapper(
                        model_name=qwen_cfg.get("model_name", "Qwen/Qwen2.5-7B-Instruct"),
                        quantization=qwen_cfg.get("quantization", "awq"),
                        max_new_tokens=qwen_cfg.get("max_new_tokens", 512),
                        device=qwen_cfg.get("device", "cuda"),
                        mc_dropout=qwen_cfg.get("mc_dropout", True),
                    )
                    llm_fn = self._qwen
            else:
                llm_fn = self._default_llm_fn

        self.grading = GradingPipeline(
            llm_fn=llm_fn,
            config=grad_cfg,
        )
        self.catalog = self.grading.catalog

        # 共享 SBERT 编码器给 RAG（避免重复加载模型）
        if encode_fn is not None:
            self.grading.retriever._encode_fn = encode_fn

    def inject_seeds(self, seed_cases: List[Dict]):
        """RAG 冷启动：注入种子案例"""
        self.grading.retriever.inject_seed_cases(seed_cases)

    def process(
        self,
        file_path: str,
        profile_result: ProfileMatchResult,
        metadata: Dict = None,
    ) -> GradingResult:
        """认知层处理：敏感数据分级

        输入：感知层传来的 ProfileMatchResult
        输出：GradingResult 传递给执行层

        Returns:
            GradingResult
        """
        grading = self.grading.grade_document(
            file_path=file_path,
            content_snippet=profile_result.content_text[:2000],
            profile=profile_result.user_profile,
            metadata=metadata,
        )

        # 置信度不足时等级升格（宁可误报不可漏报）
        effective_level = grading.level
        if grading.confidence < 0.7 and grading.level in ("L1", "L2", "L3"):
            level_map = {"L1": "L2", "L2": "L3", "L3": "L4"}
            effective_level = level_map[grading.level]

        return GradingResult(
            file_path=file_path,
            level=grading.level,
            effective_level=effective_level,
            confidence=grading.confidence,
            is_accepted=grading.is_accepted,
            needs_human_review=grading.needs_human_review,
            correlation_id=profile_result.correlation_id,
        )

    @staticmethod
    def _default_llm_fn(prompt: str, seed: int = 0) -> str:
        import json, random
        random.seed(seed)
        level = random.choice(["L1", "L2", "L3", "L4"])
        conf = round(random.uniform(0.6, 0.95), 2)
        return json.dumps({
            "level": level, "category": "auto",
            "reason": f"seed={seed}", "confidence": conf,
        })


# =====================================================================
# 执行层服务
# =====================================================================

class ExecutionService:
    """执行层服务（第4章）

    职责：
    1. 接收 ProfileMatchResult + GradingResult
    2. 三维偏离评估（主体一致性 / 行为偏离 / 环境可信度）
    3. 风险融合 → 三级响应决策（放行 / 告警 / 阻断）
    4. 为告警和阻断事件生成 VQL 检测规则（四元组 IR → VQL）
    5. 输出 DetectionVerdict + RuleDeployment

    订阅事件：GRADING_DONE, CATALOG_UPDATED
    发布事件：DECISION_BLOCK, DECISION_ALERT, DECISION_PASS, RULE_GENERATED, RULE_PUSHED
    """

    def __init__(self, config: Dict):
        det_cfg = config.get("detection", {})
        adaptive_cfg = det_cfg.get("adaptive_decision", {})
        self.decision_engine = AdaptiveDecisionEngine(
            alpha=adaptive_cfg.get("risk_weights", {}).get("alpha", 0.35),
            beta=adaptive_cfg.get("risk_weights", {}).get("beta", 0.40),
            gamma=adaptive_cfg.get("risk_weights", {}).get("gamma", 0.25),
            theta_2=adaptive_cfg.get("thresholds", {}).get("theta_2", 0.7),
            eta=adaptive_cfg.get("profile_baseline", {}).get("eta", 0.15),
        )
        self.template_synthesizer = TemplateSynthesizer()
        self.vql_compiler = VQLCompiler()

    def set_baseline(self, user_id: str, baseline: ProfileBaseline):
        self.decision_engine.set_baseline(user_id, baseline)

    def process(
        self,
        event: OperationEvent,
        profile_result: ProfileMatchResult,
        grading_result: GradingResult,
        owner_profile: np.ndarray,
    ) -> Tuple[DetectionVerdict, Optional[RuleDeployment]]:
        """执行层处理：风险决策 + 规则生成

        输入：感知层 ProfileMatchResult + 认知层 GradingResult
        输出：DetectionVerdict + 可选 RuleDeployment

        Returns:
            (verdict, rule_deployment)
        """
        t_start = time.time()

        # === 三维偏离评估 + 风险融合 ===
        decision = self.decision_engine.decide(
            event=event,
            operator_profile=profile_result.feature_vector,
            owner_profile=owner_profile,
            behavior_feature=profile_result.behavior_pattern,
        )

        # 敏感度增强
        sensitivity_boost = {"L1": 0.0, "L2": 0.05, "L3": 0.15, "L4": 0.25}
        boosted_risk = min(
            decision.risk_score + sensitivity_boost.get(grading_result.effective_level, 0.0),
            1.0,
        )

        # 阈值决策
        if boosted_risk >= self.decision_engine.theta_2:
            final_action = ResponseAction.BLOCK
        elif boosted_risk >= self.decision_engine.theta_1:
            final_action = ResponseAction.ALERT
        else:
            final_action = decision.action

        # === VQL 规则生成（仅告警/阻断时触发） ===
        vql_script = ""
        rule_deployment = None

        if final_action != ResponseAction.SILENT_MONITOR:
            ir = QuadrupleIR.from_natural_language(
                f"user_role={profile_result.user_profile.role} "
                f"process={event.process_name} "
                f"level={grading_result.effective_level} "
                f"action={event.action}"
            )
            vql_result = self.template_synthesizer.synthesize(ir)
            if vql_result:
                vql_script = vql_result
                rule_deployment = RuleDeployment(
                    rule_id=f"rule_{uuid.uuid4().hex[:8]}",
                    vql_script=vql_script,
                    target_role=profile_result.user_profile.role,
                    sensitivity_level=grading_result.effective_level,
                    response_action=final_action.value,
                    correlation_id=profile_result.correlation_id,
                )

        # === 组装裁决 ===
        t_end = time.time()
        verdict = DetectionVerdict(
            response_action=final_action.value,
            risk_score=boosted_risk,
            sensitivity_level=grading_result.effective_level,
            grading_confidence=grading_result.confidence,
            grading_accepted=grading_result.is_accepted,
            needs_human_review=grading_result.needs_human_review,
            user_id=profile_result.user_id,
            user_cluster=profile_result.cluster_id,
            profile_version=profile_result.profile_version,
            subject_consistency=decision.subject_consistency,
            behavior_deviation=decision.behavior_deviation,
            env_confidence=decision.env_confidence,
            vql_script=vql_script,
            file_path=event.file_path,
            action=event.action,
            latency_ms=(t_end - t_start) * 1000,
            reason=self._build_reason(decision, grading_result, final_action),
        )

        return verdict, rule_deployment

    @staticmethod
    def _build_reason(decision, grading, final_action) -> str:
        parts = [f"level={grading.effective_level}(conf={grading.confidence:.2f})"]
        if grading.needs_human_review:
            parts.append("needs_review")
        parts.append(f"s_u={decision.subject_consistency:.2f}")
        parts.append(f"d_b={decision.behavior_deviation:.2f}")
        parts.append(f"c_e={decision.env_confidence:.2f}")
        action_str = final_action.value if isinstance(final_action, ResponseAction) else str(final_action)
        parts.append(f"-> {action_str}")
        return "; ".join(parts)


# =====================================================================
# 系统服务端
# =====================================================================

class DLPServer:
    """终端数据泄露检测系统 — 服务端

    统一管理三个独立服务模块，通过事件总线进行模块间数据传递。
    对外提供 process_file_event() 作为事件接入点，
    内部由三层服务依次处理、各自输出结构化数据对象。

    部署模式：
    1. 单机部署 — 三层服务在同一进程，event_bus 进程内通信
    2. 分布式部署 — 各服务独立进程，event_bus 替换为消息队列

    与外部系统的集成：
    · TerminalAgent  → 通过 event_bus 接收 FILE_ACCESS 事件
    · Velociraptor   → 通过 VelociraptorBridge gRPC 下发 VQL 规则
    · 安全运营平台   → 通过 event_bus 订阅 DECISION_ALERT/BLOCK 事件

    典型调用：
        server = DLPServer.from_config("configs/default_config.yaml")
        server.init(users_meta=..., seed_cases=...)

        verdict = server.process_file_event(
            file_path="C:/Users/bob/Q1报表.xlsx",
            file_content=open(..., "rb").read(),
            user_id="vm2_bob",
            action="upload",
            process_name="chrome.exe",
            network_env="external",
        )
    """

    def __init__(
        self,
        config: Dict[str, Any] = None,
        llm_fn: Callable = None,
        device: str = "cuda",
    ):
        self.config = config or {}
        self.device = device

        # ---- 用户注册表 ----
        self._users: Dict[str, UserRecord] = {}

        # ---- 事件总线（模块间数据传递通道） ----
        self.event_bus = create_default_bus()

        # ---- 三个独立服务模块 ----
        self.perception = PerceptionService(self.config, device)
        # 共享 SBERT 编码器给认知层 RAG，避免重复加载模型
        sbert_encode_fn = self.perception.doc_parser._sbert_encode
        self.cognition = CognitionService(self.config, llm_fn, encode_fn=sbert_encode_fn)
        self.execution = ExecutionService(self.config)

        # ---- Velociraptor 桥接 ----
        self.velo_bridge: Optional[VelociraptorBridge] = None
        velo_cfg = self.config.get("velociraptor", {})
        if _HAS_VELO_BRIDGE and velo_cfg.get("enabled", False):
            self.velo_bridge = VelociraptorBridge(VelociraptorConfig(
                api_config_path=velo_cfg.get("api_config", ""),
                server_address=velo_cfg.get("server_address", "localhost:8001"),
            ))
            self.velo_bridge.connect()

        # ---- 注册服务间事件订阅 ----
        self._wire_services()
        self._initialized = False

    def _wire_services(self):
        """注册各服务模块的事件订阅（模块间数据传递路由）"""

        # 认知层订阅：感知层画像匹配完成 → 启动分级
        self.event_bus.subscribe(
            EventType.PROFILE_MATCHED,
            self._on_profile_matched,
            subscriber_name="cognition_service",
        )

        # 执行层订阅：分级完成 → 启动风险决策
        self.event_bus.subscribe(
            EventType.GRADING_DONE,
            self._on_grading_done,
            subscriber_name="execution_service",
        )

        # 执行层订阅：资产目录变更 → 更新检测规则
        self.event_bus.subscribe(
            EventType.CATALOG_UPDATED,
            self._on_catalog_updated,
            subscriber_name="execution_service",
        )

        # 感知层反馈：画像漂移 → 触发增量更新
        self.event_bus.subscribe(
            EventType.PROFILE_DRIFT,
            self._on_profile_drift,
            subscriber_name="perception_service",
        )

        # 资产目录变更监听
        self.cognition.catalog.add_change_listener(self._on_sensitivity_change)

    # -----------------------------------------------------------------
    # 事件回调（模块间数据流）
    # -----------------------------------------------------------------

    def _on_profile_matched(self, event: Event):
        """感知层 → 认知层：画像匹配完成，传递 ProfileMatchResult"""
        result = event.data.get("profile_result")
        if result:
            print(f"  [EventBus] PROFILE_MATCHED → CognitionService "
                  f"(user={result.user_id}, cluster={result.cluster_id})")

    def _on_grading_done(self, event: Event):
        """认知层 → 执行层：分级完成，传递 GradingResult"""
        result = event.data.get("grading_result")
        if result:
            print(f"  [EventBus] GRADING_DONE → ExecutionService "
                  f"(level={result.effective_level}, conf={result.confidence:.2f})")

    def _on_catalog_updated(self, event: Event):
        """认知层 → 执行层：资产目录变更，触发规则更新"""
        print(f"  [EventBus] CATALOG_UPDATED → ExecutionService (规则需更新)")

    def _on_profile_drift(self, event: Event):
        """执行层 → 感知层：画像漂移反馈"""
        user_id = event.data.get("user_id", "")
        print(f"  [EventBus] PROFILE_DRIFT → PerceptionService (user={user_id})")

    def _on_sensitivity_change(self, change_event: Dict):
        """资产敏感度变更回调（ch3 → ch4 联动）"""
        self.event_bus.publish(Event(
            event_type=EventType.CATALOG_UPDATED,
            source="cognition_service",
            data=change_event,
        ))

    # -----------------------------------------------------------------
    # 工厂方法
    # -----------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str, llm_fn: Callable = None, device: str = "cuda"):
        """从 YAML 配置文件创建服务端"""
        import yaml

        merged = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                merged = yaml.safe_load(f) or {}

        base_dir = os.path.dirname(config_path)
        for sub in ["profiling_config.yaml", "grading_config.yaml", "detection_config.yaml"]:
            sub_path = os.path.join(base_dir, sub)
            if os.path.exists(sub_path):
                with open(sub_path, "r", encoding="utf-8") as f:
                    sub_cfg = yaml.safe_load(f) or {}
                key = sub.replace("_config.yaml", "")
                merged[key] = sub_cfg

        return cls(config=merged, llm_fn=llm_fn, device=device)

    # -----------------------------------------------------------------
    # 系统初始化
    # -----------------------------------------------------------------

    def register_user(self, user_meta: Dict[str, Any]):
        uid = user_meta["user_id"]
        self._users[uid] = UserRecord(
            user_id=uid,
            role=user_meta.get("role", "default"),
            department=user_meta.get("department", ""),
            projects=user_meta.get("projects", []),
            permissions=user_meta.get("permissions", []),
            business_keywords=user_meta.get("business_keywords", []),
        )

    def init(
        self,
        users_meta: List[Dict[str, Any]],
        seed_cases: List[Dict] = None,
        historical_features: np.ndarray = None,
    ):
        """系统初始化"""
        # 1. 注册用户
        for meta in users_meta:
            self.register_user(meta)
        print(f"[init] {len(self._users)} users registered")

        # 2. 认知层冷启动
        if seed_cases:
            self.cognition.inject_seeds(seed_cases)
            print(f"[init] {len(seed_cases)} RAG seed cases injected")

        # 3. 感知层初始聚类
        if historical_features is not None:
            cluster_result = self.perception.init_clusters(historical_features)
            user_ids = list(self._users.keys())
            for i, uid in enumerate(user_ids[:len(cluster_result.assignments)]):
                cid = int(cluster_result.assignments[i])
                self._users[uid].cluster_id = cid
                self._users[uid].feature_vector = historical_features[i]

                # 初始化执行层基线
                self.execution.set_baseline(uid, ProfileBaseline(
                    mean=cluster_result.centers[cid],
                    covariance=np.eye(historical_features.shape[1]) * 0.1,
                ))
            sc = IGWKMeans.silhouette_coefficient(
                historical_features, cluster_result.assignments
            )
            print(f"[init] IGW-Kmeans clustering done: K={self.perception.igw_kmeans.K}, SC={sc:.3f}")

        # 4. 发布系统初始化事件
        self.event_bus.publish(Event(
            event_type=EventType.SYSTEM_INIT,
            source="dlp_server",
            data={"user_count": len(self._users)},
        ))
        self._initialized = True
        print("[init] system ready")

    # -----------------------------------------------------------------
    # 核心：处理文件事件（三层模块协作）
    # -----------------------------------------------------------------

    def process_file_event(
        self,
        file_path: str,
        file_content: bytes,
        user_id: str,
        action: str = "read",
        process_name: str = "explorer.exe",
        device_type: str = "local",
        network_env: str = "internal",
        is_work_hours: bool = True,
        metadata: Dict[str, Any] = None,
    ) -> DetectionVerdict:
        """处理单个文件操作事件

        数据在三层模块之间的传递过程：

        ① TerminalAgent/外部 → 本方法：原始文件事件
        ② 本方法 → PerceptionService.process()
           输出 ProfileMatchResult → 广播 PROFILE_MATCHED 事件
        ③ ProfileMatchResult → CognitionService.process()
           输出 GradingResult → 广播 GRADING_DONE 事件
        ④ ProfileMatchResult + GradingResult → ExecutionService.process()
           输出 (DetectionVerdict, RuleDeployment)
        ⑤ RuleDeployment → VelociraptorBridge → 终端 VQL 执行
           广播 DECISION_* 事件
        """
        t_start = time.time()
        correlation_id = uuid.uuid4().hex[:12]

        # === 用户查找 ===
        user = self._users.get(user_id)
        if user is None:
            user = UserRecord(
                user_id=user_id, role="unknown", department="unknown",
                projects=[], permissions=[], business_keywords=[],
            )
            self._users[user_id] = user

        # ============================================================
        # ① 感知层处理 → 输出 ProfileMatchResult
        # ============================================================
        profile_result = self.perception.process(
            file_content=file_content,
            file_path=file_path,
            user=user,
            action=action,
            process_name=process_name,
            device_type=device_type,
            network_env=network_env,
            is_work_hours=is_work_hours,
            correlation_id=correlation_id,
        )

        # 广播：感知层 → 认知层 / 执行层
        self.event_bus.publish(Event(
            event_type=EventType.PROFILE_MATCHED,
            source="perception_service",
            data={"profile_result": profile_result},
            correlation_id=correlation_id,
        ))

        # ============================================================
        # ② 认知层处理 → 输出 GradingResult
        # ============================================================
        grading_result = self.cognition.process(
            file_path=file_path,
            profile_result=profile_result,
            metadata=metadata,
        )

        # 广播：认知层 → 执行层
        grading_event_type = EventType.GRADING_DONE
        if grading_result.needs_human_review:
            grading_event_type = EventType.GRADING_REVIEW
        if grading_result.effective_level != grading_result.level:
            grading_event_type = EventType.GRADING_ESCALATED

        self.event_bus.publish(Event(
            event_type=grading_event_type,
            source="cognition_service",
            data={"grading_result": grading_result},
            correlation_id=correlation_id,
        ))

        # ============================================================
        # ③ 执行层处理 → 输出 DetectionVerdict + RuleDeployment
        # ============================================================
        event = OperationEvent(
            user_id=user_id,
            action=action,
            file_path=file_path,
            process_name=process_name,
            device_type=device_type,
            network_env=network_env,
            is_work_hours=is_work_hours,
        )

        # 资产所有者画像
        owner_profile = self._get_owner_profile(file_path, grading_result.effective_level)

        verdict, rule_deployment = self.execution.process(
            event=event,
            profile_result=profile_result,
            grading_result=grading_result,
            owner_profile=owner_profile,
        )

        # 补充端到端延迟
        verdict.latency_ms = (time.time() - t_start) * 1000

        # ============================================================
        # ④ 规则下发 → Velociraptor / 终端代理
        # ============================================================
        if rule_deployment:
            self._deploy_rule(rule_deployment)

        # ============================================================
        # ⑤ 广播决策结果
        # ============================================================
        decision_type_map = {
            "block": EventType.DECISION_BLOCK,
            "alert": EventType.DECISION_ALERT,
            "silent_monitoring": EventType.DECISION_PASS,
        }
        self.event_bus.publish(Event(
            event_type=decision_type_map.get(verdict.response_action, EventType.DECISION_PASS),
            source="execution_service",
            data={
                "verdict": verdict,
                "rule_deployment": rule_deployment,
            },
            correlation_id=correlation_id,
        ))

        return verdict

    # -----------------------------------------------------------------
    # 规则下发
    # -----------------------------------------------------------------

    def _deploy_rule(self, rule: RuleDeployment):
        """将执行层生成的 VQL 规则下发到 Velociraptor 或终端代理"""
        # Velociraptor 在线 → gRPC 下发
        if self.velo_bridge and self.velo_bridge.is_connected:
            try:
                artifact = self.velo_bridge.create_dlp_artifact(
                    rule_name=rule.rule_id,
                    vql_script=rule.vql_script,
                    user_role=rule.target_role,
                    sensitivity_level=rule.sensitivity_level,
                    response_action=rule.response_action,
                )
                self.velo_bridge.register_artifact(artifact)
                rule.artifact_yaml = artifact.to_yaml()
                rule.deployed = True
            except Exception as e:
                print(f"[deploy] Velociraptor rule deploy failed: {e}")

        # 广播规则下发事件（终端代理可订阅）
        self.event_bus.publish(Event(
            event_type=EventType.RULE_PUSHED,
            source="execution_service",
            data={
                "rule_id": rule.rule_id,
                "vql_script": rule.vql_script,
                "target_role": rule.target_role,
                "deployed_to_velociraptor": rule.deployed,
            },
            correlation_id=rule.correlation_id,
        ))

    # -----------------------------------------------------------------
    # 辅助
    # -----------------------------------------------------------------

    def _get_owner_profile(self, file_path: str, level: str) -> np.ndarray:
        if self.perception.cluster_result is not None:
            K = self.perception.igw_kmeans.K
            level_cluster = {"L1": 0, "L2": 0, "L3": 1, "L4": min(2, K - 1)}
            cid = level_cluster.get(level, 0)
            return self.perception.cluster_result.centers[cid]
        return np.zeros(896)

    def process_batch(self, events: List[Dict[str, Any]]) -> List[DetectionVerdict]:
        return [self.process_file_event(**ev) for ev in events]

    def get_event_chain(self, correlation_id: str) -> List[Event]:
        """追踪一个文件事件的完整处理链路"""
        return self.event_bus.get_processing_chain(correlation_id)

    def get_system_stats(self) -> Dict:
        return {
            "users": len(self._users),
            "initialized": self._initialized,
            "velociraptor_connected": (
                self.velo_bridge.is_connected if self.velo_bridge else False
            ),
            "event_bus": self.event_bus.get_stats(),
        }


# =====================================================================
# 向后兼容别名
# =====================================================================
DLPPipeline = DLPServer
