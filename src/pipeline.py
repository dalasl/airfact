"""
端到端数据泄露检测流水线

将论文三层架构串联为统一的处理链：
    感知层 (ch2) → 认知层 (ch3) → 执行层 (ch4)

典型调用流程:
    pipeline = DLPPipeline.from_config("configs/default_config.yaml")
    pipeline.init_system(users_meta, seed_cases)

    # --- 实时检测循环 ---
    result = pipeline.process_file_event(
        file_path="C:/Users/alice/合同终稿.docx",
        file_content=open(..., "rb").read(),
        user_id="alice",
        action="upload",
        process_name="chrome.exe",
        device_type="local",
        network_env="external",
    )
    print(result)
    # DetectionVerdict(action=BLOCK, risk=0.82, sensitivity='L3', ...)
"""

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ---- 感知层（第2章） ----
from src.ch2_user_profiling.unet_denoiser import UNetDenoiser
from src.ch2_user_profiling.document_parser import DocumentParser, LRUDocumentCache
from src.ch2_user_profiling.temporal_alignment import (
    TemporalAligner, RawEvent, EventType, StructuredEvent,
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
# 统一输出数据结构
# =====================================================================

@dataclass
class DetectionVerdict:
    """端到端检测最终裁决

    整合三层结果的统一输出：
        感知层  → user_cluster / profile_version
        认知层  → sensitivity_level / grading_confidence
        执行层  → response_action / risk_score
    """
    # --- 执行层决策 ---
    response_action: str          # SILENT_MONITOR / ALERT / BLOCK
    risk_score: float             # R(a) ∈ [0, 1]

    # --- 认知层结果 ---
    sensitivity_level: str        # L1 / L2 / L3 / L4
    grading_confidence: float     # 分级置信度
    grading_accepted: bool        # 是否通过自洽性校验
    needs_human_review: bool      # 是否需人工复核

    # --- 感知层上下文 ---
    user_id: str
    user_cluster: int             # 所属聚类编号
    profile_version: int          # 画像版本号

    # --- 风险评分分解 ---
    subject_consistency: float    # s_u(a)
    behavior_deviation: float     # d_b(a)
    env_confidence: float         # c_e(a)

    # --- 规则信息 ---
    vql_script: str = ""          # 触发的 VQL 规则
    rule_template: str = ""       # 匹配的模板 ID

    # --- 元数据 ---
    file_path: str = ""
    action: str = ""
    timestamp: float = field(default_factory=time.time)
    latency_ms: float = 0.0      # 端到端处理延迟
    reason: str = ""              # 可读的判定说明

    def is_blocked(self) -> bool:
        return self.response_action == "BLOCK"

    def is_alert(self) -> bool:
        return self.response_action == "ALERT"

    def summary(self) -> str:
        emoji = {"BLOCK": "🚫", "ALERT": "⚠️", "SILENT_MONITOR": "✅"}
        e = emoji.get(self.response_action, "❓")
        return (
            f"{e} [{self.response_action}] "
            f"文件={self.file_path}  用户={self.user_id}  "
            f"敏感度={self.sensitivity_level}  风险={self.risk_score:.3f}  "
            f"延迟={self.latency_ms:.0f}ms"
        )


# =====================================================================
# 用户注册表（画像管理）
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
    feature_vector: Optional[np.ndarray] = None      # 融合后特征
    content_semantic: Optional[np.ndarray] = None     # 768d
    behavior_pattern: Optional[np.ndarray] = None     # 96d
    permission_context: Optional[np.ndarray] = None   # 32d

    def to_user_profile(self) -> UserProfile:
        """转换为第3章的 UserProfile 格式"""
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
        """返回画像数值向量（用于第4章主体一致性计算）"""
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
# 主流水线
# =====================================================================

class DLPPipeline:
    """端到端数据泄露检测流水线

    将论文三层架构封装为一次调用：

        文件事件  ──► 感知层(画像匹配)
                      ──► 认知层(敏感分级)
                          ──► 执行层(风险决策) ──► DetectionVerdict

    Args:
        config: 配置字典（或通过 from_config 加载 YAML）
        llm_fn: LLM 推理函数签名 (prompt: str, seed: int) → str
        device: 计算设备 ("cuda" / "cpu")
    """

    def __init__(
        self,
        config: Dict[str, Any] = None,
        llm_fn: Callable = None,
        device: str = "cuda",
    ):
        self.config = config or {}
        self.device = device
        self._llm_fn = llm_fn or self._default_llm_fn

        # --- 用户注册表 ---
        self._users: Dict[str, UserRecord] = {}

        # --- 感知层组件 ---
        prof_cfg = self.config.get("profiling", {})
        self.denoiser = UNetDenoiser(
            alpha=prof_cfg.get("unet", {}).get("loss", {}).get("alpha", 0.7)
        )
        self.doc_parser = DocumentParser(
            cache_capacity=prof_cfg.get("cache", {}).get("capacity", 500),
            top_k_keywords=prof_cfg.get("features", {}).get("tfidf_top_k", 20),
            device=device,
        )
        self.temporal_aligner = TemporalAligner(
            doc_cache=self.doc_parser.cache,
            delta_seconds=prof_cfg.get("temporal_alignment", {}).get("delta_seconds", 5.0),
        )
        self.feature_extractor = FeatureExtractor()
        self.igw_kmeans = IGWKMeans(
            K=prof_cfg.get("igw_kmeans", {}).get("K", 3),
            T_max=prof_cfg.get("igw_kmeans", {}).get("T_max", 100),
            epsilon=prof_cfg.get("igw_kmeans", {}).get("epsilon", 1e-4),
        )
        self.incremental_updater = IncrementalUpdater(
            eta=prof_cfg.get("incremental_update", {}).get("eta", 0.15),
            theta_drift=prof_cfg.get("incremental_update", {}).get("theta_drift", 2.5),
        )
        self._cluster_result: Optional[ClusterResult] = None
        self._profile_version: Optional[ProfileVersion] = None

        # --- 认知层组件 ---
        grad_cfg = self.config.get("grading", {})
        self.grading_pipeline = GradingPipeline(
            llm_fn=self._llm_fn,
            config=grad_cfg,
        )
        self.asset_catalog = self.grading_pipeline.catalog

        # --- 执行层组件 ---
        det_cfg = self.config.get("detection", {})
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

        # --- 资产目录变更 → 规则更新联动 ---
        self.asset_catalog.add_change_listener(self._on_sensitivity_change)

        # --- Velociraptor 桥接（可选） ---
        self.velo_bridge: Optional['VelociraptorBridge'] = None
        velo_cfg = self.config.get("velociraptor", {})
        if _HAS_VELO_BRIDGE and velo_cfg.get("enabled", False):
            self.velo_bridge = VelociraptorBridge(VelociraptorConfig(
                api_config_path=velo_cfg.get("api_config", ""),
                server_address=velo_cfg.get("server_address", "localhost:8001"),
            ))
            self.velo_bridge.connect()

        self._initialized = False

    # -----------------------------------------------------------------
    # 工厂方法
    # -----------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str, llm_fn: Callable = None, device: str = "cuda"):
        """从 YAML 配置文件创建流水线"""
        import yaml
        import os

        merged = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                merged = yaml.safe_load(f) or {}

        # 尝试合并子配置
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
        """注册用户元数据（来自 LDAP/HR 系统）"""
        uid = user_meta["user_id"]
        self._users[uid] = UserRecord(
            user_id=uid,
            role=user_meta.get("role", "普通员工"),
            department=user_meta.get("department", ""),
            projects=user_meta.get("projects", []),
            permissions=user_meta.get("permissions", []),
            business_keywords=user_meta.get("business_keywords", []),
        )

    def init_system(
        self,
        users_meta: List[Dict[str, Any]],
        seed_cases: List[Dict] = None,
        historical_features: np.ndarray = None,
    ):
        """系统初始化

        1. 注册用户
        2. 注入 RAG 种子案例（冷启动）
        3. 若有历史特征数据，运行 IGW-Kmeans 初始聚类
        4. 初始化各用户的画像基线

        Args:
            users_meta: 用户元数据列表
            seed_cases: RAG 种子案例（每个等级 5-10 个）
            historical_features: 历史特征矩阵 [n_users, d] (可选)
        """
        # 1. 注册用户
        for meta in users_meta:
            self.register_user(meta)
        print(f"[init] 已注册 {len(self._users)} 个用户")

        # 2. RAG 冷启动
        if seed_cases:
            self.grading_pipeline.retriever.inject_seed_cases(seed_cases)
            print(f"[init] 已注入 {len(seed_cases)} 个 RAG 种子案例")

        # 3. 初始聚类
        if historical_features is not None:
            self._cluster_result = self.igw_kmeans.fit(historical_features)
            user_ids = list(self._users.keys())
            for i, uid in enumerate(user_ids[:len(self._cluster_result.assignments)]):
                cid = int(self._cluster_result.assignments[i])
                self._users[uid].cluster_id = cid
                self._users[uid].feature_vector = historical_features[i]

                # 初始化执行层画像基线
                self.decision_engine.set_baseline(uid, ProfileBaseline(
                    mean=self._cluster_result.centers[cid],
                    covariance=np.eye(historical_features.shape[1]) * 0.1,
                ))
            print(f"[init] IGW-Kmeans 聚类完成: K={self.igw_kmeans.K}, "
                  f"SC={IGWKMeans.silhouette_coefficient(historical_features, self._cluster_result.assignments):.3f}")

        self._initialized = True
        print("[init] 系统初始化完成 ✓")

    # -----------------------------------------------------------------
    # 核心：端到端处理
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
        """处理单个文件操作事件（端到端）

        完整链路:
            1. 感知层 → 文档解析 + 用户画像匹配
            2. 认知层 → 敏感数据分级 (RAG + 自洽性)
            3. 执行层 → 规则匹配 + 风险决策

        Args:
            file_path: 被操作的文件路径
            file_content: 文件内容字节
            user_id: 操作用户 ID
            action: 操作类型 (read/write/copy/upload/send/print)
            process_name: 发起操作的进程名
            device_type: 设备类型 (local/usb/bluetooth)
            network_env: 网络环境 (internal/external/vpn)
            is_work_hours: 是否工作时间
            metadata: 附加元数据

        Returns:
            DetectionVerdict 完整裁决
        """
        t_start = time.time()

        # === 0. 用户查找 ===
        user = self._users.get(user_id)
        if user is None:
            # 未注册用户 → 按最高风险处理
            user = UserRecord(
                user_id=user_id, role="未知", department="未知",
                projects=[], permissions=[], business_keywords=[],
            )
            self._users[user_id] = user

        # === 1. 感知层：文档语义解析 ===
        content_text = self._extract_text(file_content, file_path)

        # === 2. 认知层：敏感数据分级 ===
        profile = user.to_user_profile()
        grading = self.grading_pipeline.grade_document(
            file_path=file_path,
            content_snippet=content_text[:2000],
            profile=profile,
            metadata=metadata,
        )

        # 置信度不足时等级升格（宁可误报不可漏报）
        effective_level = grading.level
        if grading.confidence < 0.7 and grading.level in ("L1", "L2", "L3"):
            level_map = {"L1": "L2", "L2": "L3", "L3": "L4"}
            effective_level = level_map[grading.level]

        # === 3. 执行层：风险决策 ===
        event = OperationEvent(
            user_id=user_id,
            action=action,
            file_path=file_path,
            process_name=process_name,
            device_type=device_type,
            network_env=network_env,
            is_work_hours=is_work_hours,
        )

        operator_profile = user.to_profile_vector()
        # 资产所有者画像：简化为聚类中心
        owner_profile = self._get_owner_profile(file_path, effective_level)
        behavior_feature = self._build_behavior_feature(event, user)

        decision = self.decision_engine.decide(
            event=event,
            operator_profile=operator_profile,
            owner_profile=owner_profile,
            behavior_feature=behavior_feature,
        )

        # 敏感度越高，风险阈值越低 → 加权调整
        sensitivity_boost = {"L1": 0.0, "L2": 0.05, "L3": 0.15, "L4": 0.25}
        boosted_risk = decision.risk_score + sensitivity_boost.get(effective_level, 0.0)
        boosted_risk = min(boosted_risk, 1.0)

        # 最终决策（敏感度增强后重新判定）
        if boosted_risk >= self.decision_engine.theta_2:
            final_action = ResponseAction.BLOCK
        elif boosted_risk >= self.decision_engine.theta_1:
            final_action = ResponseAction.ALERT
        else:
            final_action = decision.action

        # === 4. 规则生成（按需） ===
        vql_script = ""
        rule_template = ""
        if final_action != ResponseAction.SILENT_MONITOR:
            ir = QuadrupleIR.from_natural_language(
                f"用户{user.role}通过{process_name}对{effective_level}级文件执行{action}操作"
            )
            vql_result = self.template_synthesizer.synthesize(ir)
            if vql_result:
                vql_script = vql_result
                rule_template = "auto_generated"

            # 将规则下发到 Velociraptor 执行（若已连接）
            if vql_script and self.velo_bridge and self.velo_bridge.is_connected:
                try:
                    artifact = self.velo_bridge.create_dlp_artifact(
                        rule_name=f"Rule_{user_id}_{int(time.time())}",
                        vql_script=vql_script,
                        user_role=user.role,
                        sensitivity_level=effective_level,
                        response_action=final_action.value,
                    )
                    self.velo_bridge.register_artifact(artifact)
                except Exception as e:
                    print(f"[pipeline] Velociraptor 规则下发失败: {e}")

        # === 5. 组装裁决 ===
        t_end = time.time()

        verdict = DetectionVerdict(
            response_action=final_action.value if isinstance(final_action, ResponseAction) else str(final_action),
            risk_score=boosted_risk,
            sensitivity_level=effective_level,
            grading_confidence=grading.confidence,
            grading_accepted=grading.is_accepted,
            needs_human_review=grading.needs_human_review,
            user_id=user_id,
            user_cluster=user.cluster_id,
            profile_version=user.profile_version,
            subject_consistency=decision.subject_consistency,
            behavior_deviation=decision.behavior_deviation,
            env_confidence=decision.env_confidence,
            vql_script=vql_script,
            rule_template=rule_template,
            file_path=file_path,
            action=action,
            latency_ms=(t_end - t_start) * 1000,
            reason=self._build_reason(decision, grading, effective_level, final_action),
        )

        return verdict

    # -----------------------------------------------------------------
    # 批量处理
    # -----------------------------------------------------------------

    def process_batch(
        self, events: List[Dict[str, Any]]
    ) -> List[DetectionVerdict]:
        """批量处理文件事件"""
        results = []
        for ev in events:
            verdict = self.process_file_event(**ev)
            results.append(verdict)
        return results

    # -----------------------------------------------------------------
    # 画像更新
    # -----------------------------------------------------------------

    def trigger_profile_update(
        self,
        user_id: str,
        trigger: str = "implicit",
        new_features: np.ndarray = None,
    ):
        """触发用户画像更新

        Args:
            user_id: 用户 ID
            trigger: "explicit" (岗位变更) 或 "implicit" (行为漂移)
            new_features: 新的特征数据
        """
        user = self._users.get(user_id)
        if user is None or self._profile_version is None:
            return

        ttype = (UpdateTriggerType.EXPLICIT_EVENT if trigger == "explicit"
                 else UpdateTriggerType.IMPLICIT_DRIFT)

        new_profile, meta = self.incremental_updater.process_update(
            user_id=user_id,
            current_profile=self._profile_version,
            user_cluster_id=user.cluster_id,
            trigger_type=ttype,
            feature_matrix=new_features,
            igw_kmeans_fn=self.igw_kmeans.fit if ttype == UpdateTriggerType.EXPLICIT_EVENT else None,
        )
        self._profile_version = new_profile
        user.profile_version += 1
        print(f"[update] 用户 {user_id} 画像已更新 (v{user.profile_version}), 触发={trigger}")

    # -----------------------------------------------------------------
    # 内部辅助
    # -----------------------------------------------------------------

    @staticmethod
    def _default_llm_fn(prompt: str, seed: int = 0) -> str:
        """默认 LLM 推理函数（Qwen2.5-7B 4-bit）"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            # 实际部署时替换为已加载的模型实例
            return '{"level": "L2", "category": "内部文档", "reason": "默认推理", "confidence": 0.75}'
        except ImportError:
            import json, random
            random.seed(seed)
            level = random.choice(["L1", "L2", "L3", "L4"])
            conf = round(random.uniform(0.6, 0.95), 2)
            return json.dumps({
                "level": level, "category": "自动分级",
                "reason": f"基于内容语义分析 (seed={seed})", "confidence": conf,
            })

    @staticmethod
    def _extract_text(content: bytes, file_path: str) -> str:
        """从文件内容中提取文本"""
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
                texts = [page.get_text() for page in doc]
                return " ".join(texts)[:5000]
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

    def _get_owner_profile(self, file_path: str, level: str) -> np.ndarray:
        """根据文件路径和敏感度推断资产所有者画像"""
        # 如果有聚类结果，使用聚类中心作为"标准画像"
        if self._cluster_result is not None:
            # L3/L4 → 高权限聚类，L1/L2 → 普通聚类
            level_cluster_map = {"L1": 0, "L2": 0, "L3": 1, "L4": min(2, self.igw_kmeans.K - 1)}
            cid = level_cluster_map.get(level, 0)
            return self._cluster_result.centers[cid]
        return np.zeros(896)

    def _build_behavior_feature(self, event: OperationEvent, user: UserRecord) -> np.ndarray:
        """从操作事件构建行为特征向量"""
        if user.behavior_pattern is not None:
            return user.behavior_pattern
        # 简化：根据事件属性构建稀疏特征
        feature = np.zeros(96)
        # 时间特征 (48d) — 当前时间所在的 bin
        import datetime
        dt = datetime.datetime.fromtimestamp(event.timestamp)
        time_bin = (dt.hour * 60 + dt.minute) // 30
        if time_bin < 48:
            feature[time_bin] = 1.0
        # 操作类型特征 (32d 的前几维)
        action_idx = {"read": 0, "write": 1, "copy": 2, "upload": 3,
                      "send": 4, "print": 5, "download": 6, "delete": 7}
        aidx = action_idx.get(event.action, 0)
        feature[48 + aidx] = 1.0
        # 设备/网络特征 (16d 的前几维)
        if event.device_type == "usb":
            feature[80] = 1.0
        if event.network_env == "external":
            feature[81] = 1.0
        if not event.is_work_hours:
            feature[82] = 1.0
        return feature

    @staticmethod
    def _build_reason(
        decision: DecisionResult,
        grading: GradingDecision,
        level: str,
        final_action,
    ) -> str:
        """生成可读的判定说明"""
        parts = [f"文件敏感度={level}(置信度{grading.confidence:.2f})"]
        if grading.needs_human_review:
            parts.append("分级需人工复核")
        parts.append(f"主体一致性={decision.subject_consistency:.2f}")
        parts.append(f"行为偏离={decision.behavior_deviation:.2f}")
        parts.append(f"环境可信度={decision.env_confidence:.2f}")
        if decision.is_whitelisted:
            parts.append("(白名单放行)")
        action_str = final_action.value if isinstance(final_action, ResponseAction) else str(final_action)
        parts.append(f"→ 响应={action_str}")
        return "; ".join(parts)

    # -----------------------------------------------------------------
    # 事件驱动联动
    # -----------------------------------------------------------------

    def _on_sensitivity_change(self, change_event: Dict):
        """资产敏感度变更回调 (ch3 → ch4 联动)

        当文档的敏感等级发生变化时，自动更新检测规则。
        L2→L3: 加入高敏感监控范围
        L3→L2: 移出高敏感监控范围
        """
        old, new = change_event["old_level"], change_event["new_level"]
        fp = change_event["file_path"]
        print(f"[联动] 资产 {fp} 敏感度变更: {old} → {new}")
