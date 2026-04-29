"""
场景自适应决策引擎

算法 alg:adaptive-decision 实现。
基于画像基线的三维偏离评估 + 动态白名单 + 三级阈值响应。

三维偏离：
1. 主体一致性 s_u(a) — KL散度度量操作者与资产所有者的画像匹配度
2. 行为基线偏离 d_b(a) — Mahalanobis距离度量操作偏离日常模式的程度
3. 环境上下文可信度 c_e(a) — Sigmoid激活的环境特征评分

风险融合：R(a) = α·(1-s_u) + β·d_b + γ·(1-c_e)

三级响应：
- R < θ_1: 静默监控
- θ_1 ≤ R < θ_2: 告警提示
- R ≥ θ_2: 操作阻断

论文对应：
    - 算法 alg:adaptive-decision
    - 公式 (4.8)-(4.15)
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import expit as sigmoid
from scipy.stats import entropy as kl_divergence


class ResponseAction(Enum):
    """响应动作"""
    SILENT_MONITOR = "silent_monitoring"  # 静默监控
    ALERT = "alert"                      # 告警提示
    BLOCK = "block"                      # 操作阻断


@dataclass
class OperationEvent:
    """操作事件"""
    user_id: str
    action: str
    file_path: str
    process_name: str
    timestamp: float = field(default_factory=time.time)
    network_env: str = "internal"    # internal / external / vpn
    device_type: str = "local"       # local / usb / bluetooth
    is_work_hours: bool = True
    ip_address: str = ""


@dataclass
class ProfileBaseline:
    """画像基线（高斯过程模型）

    p(t) ~ N(μ(t), Σ(t))
    """
    mean: np.ndarray      # μ(t)
    covariance: np.ndarray  # Σ(t)
    eta: float = 0.15     # EMA 学习率


@dataclass
class DecisionResult:
    """决策结果"""
    action: ResponseAction
    risk_score: float
    subject_consistency: float  # s_u(a)
    behavior_deviation: float   # d_b(a)
    env_confidence: float       # c_e(a)
    is_whitelisted: bool = False
    reason: str = ""


class DynamicWhitelist:
    """动态白名单

    准入条件：s_u(a) > θ_wl AND d_b(a) < θ_norm
    条目带 TTL，过期后需重新评估。
    """

    def __init__(self, theta_wl: float = 0.8, theta_norm: float = 1.5, ttl_hours: float = 24):
        self.theta_wl = theta_wl
        self.theta_norm = theta_norm
        self.ttl = ttl_hours * 3600
        self._entries: Dict[str, float] = {}  # key → expiry_time

    def check(self, key: str) -> bool:
        """检查是否在白名单中"""
        if key in self._entries:
            if time.time() < self._entries[key]:
                return True
            else:
                del self._entries[key]
        return False

    def add(self, key: str):
        """添加到白名单"""
        self._entries[key] = time.time() + self.ttl

    def should_whitelist(self, s_u: float, d_b: float) -> bool:
        """判断是否满足白名单准入条件"""
        return s_u > self.theta_wl and d_b < self.theta_norm


class AdaptiveDecisionEngine:
    """场景自适应决策引擎

    Args:
        alpha: 主体一致性权重
        beta: 行为偏离权重
        gamma: 环境可信度权重
        theta_1_ratio: θ_1 = ratio × θ_2
        theta_2: 高风险阈值（通过 ROC/Youden's J 优化）
        eta: 画像基线 EMA 学习率
    """

    def __init__(
        self,
        alpha: float = 0.35,
        beta: float = 0.40,
        gamma: float = 0.25,
        theta_1_ratio: float = 0.6,
        theta_2: float = 0.7,
        eta: float = 0.15,
        whitelist_config: Dict = None,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.theta_2 = theta_2
        self.theta_1 = theta_1_ratio * theta_2
        self.eta = eta

        # 动态白名单
        wl_config = whitelist_config or {}
        self.whitelist = DynamicWhitelist(
            theta_wl=wl_config.get("theta_wl", 0.8),
            theta_norm=wl_config.get("theta_norm", 1.5),
            ttl_hours=wl_config.get("ttl_hours", 24),
        )

        # 环境评分权重（逻辑回归参数）
        self.env_weights = None

        # 用户画像基线缓存
        self._baselines: Dict[str, ProfileBaseline] = {}

    def set_baseline(self, user_id: str, baseline: ProfileBaseline):
        """设置用户画像基线"""
        self._baselines[user_id] = baseline

    def update_baseline(self, user_id: str, observation: np.ndarray):
        """在线更新画像基线（EMA）

        μ(t+1) = (1-η)μ(t) + η·x_t
        Σ(t+1) = (1-η)Σ(t) + η·[(x_t-μ(t))(x_t-μ(t))ᵀ - Σ(t)]
        """
        if user_id not in self._baselines:
            dim = len(observation)
            self._baselines[user_id] = ProfileBaseline(
                mean=observation.copy(),
                covariance=np.eye(dim) * 0.1,
                eta=self.eta,
            )
            return

        bl = self._baselines[user_id]
        eta = bl.eta

        # 均值更新
        bl.mean = (1 - eta) * bl.mean + eta * observation

        # 协方差更新
        diff = observation - bl.mean
        outer = np.outer(diff, diff)
        bl.covariance = (1 - eta) * bl.covariance + eta * (outer - bl.covariance)

    def compute_subject_consistency(
        self, operator_profile: np.ndarray, owner_profile: np.ndarray
    ) -> float:
        """维度1: 主体一致性

        s_u(a) = exp(-½ D_KL(P_subj ∥ P_owner))

        Args:
            operator_profile: 操作者画像分布
            owner_profile: 资产所有者画像分布

        Returns:
            一致性评分 ∈ (0, 1]
        """
        # 归一化为概率分布
        eps = 1e-8
        p = np.abs(operator_profile) + eps
        q = np.abs(owner_profile) + eps
        p = p / p.sum()
        q = q / q.sum()

        # KL 散度
        dkl = np.sum(p * np.log(p / q))
        return float(np.exp(-0.5 * dkl))

    def compute_behavior_deviation(
        self, behavior_feature: np.ndarray, user_id: str
    ) -> float:
        """维度2: 行为基线偏离（Mahalanobis 距离）

        d_b(a) = √((b_a - μ)ᵀ Σ⁻¹ (b_a - μ))
        """
        if user_id not in self._baselines:
            return 0.0

        bl = self._baselines[user_id]
        diff = behavior_feature - bl.mean

        # 正则化协方差
        reg_cov = bl.covariance + np.eye(bl.covariance.shape[0]) * 1e-6
        try:
            cov_inv = np.linalg.inv(reg_cov)
            dist = np.sqrt(float(diff @ cov_inv @ diff))
        except np.linalg.LinAlgError:
            dist = np.linalg.norm(diff)

        return dist

    def compute_environment_confidence(self, event: OperationEvent) -> float:
        """维度3: 环境上下文可信度

        c_e(a) = σ(wᵀ φ(a))

        Args:
            event: 操作事件

        Returns:
            可信度评分 ∈ (0, 1)
        """
        # 环境特征向量 φ(a)
        features = np.array([
            1.0 if event.network_env == "internal" else 0.0,
            1.0 if event.is_work_hours else 0.0,
            1.0 if event.device_type == "local" else 0.0,
            0.0 if event.network_env == "external" else 1.0,
        ])

        if self.env_weights is not None:
            score = self.env_weights @ features
        else:
            # 默认等权
            score = features.mean() * 3 - 1  # 映射到合理范围

        return float(sigmoid(score))

    def compute_risk_score(
        self,
        s_u: float,
        d_b: float,
        c_e: float,
    ) -> float:
        """计算复合风险评分

        R(a) = α·(1-s_u) + β·d_b + γ·(1-c_e)

        所有项与风险正相关（取补）。
        """
        # d_b 归一化（假设正常范围 0-5）
        d_b_normalized = min(d_b / 5.0, 1.0)

        return (
            self.alpha * (1 - s_u) +
            self.beta * d_b_normalized +
            self.gamma * (1 - c_e)
        )

    def decide(
        self,
        event: OperationEvent,
        operator_profile: np.ndarray,
        owner_profile: np.ndarray,
        behavior_feature: np.ndarray,
    ) -> DecisionResult:
        """完整决策流程（算法 alg:adaptive-decision）

        Args:
            event: 操作事件
            operator_profile: 操作者画像
            owner_profile: 资产所有者画像
            behavior_feature: 当前操作行为特征

        Returns:
            决策结果（含响应动作和详细评分）
        """
        # Phase 1: 基线在线更新
        self.update_baseline(event.user_id, behavior_feature)

        # Phase 2: 三维偏离计算
        s_u = self.compute_subject_consistency(operator_profile, owner_profile)
        d_b = self.compute_behavior_deviation(behavior_feature, event.user_id)
        c_e = self.compute_environment_confidence(event)

        # Phase 3: 风险评分融合
        risk = self.compute_risk_score(s_u, d_b, c_e)

        # Phase 4: 动态白名单检查
        wl_key = f"{event.user_id}:{event.file_path}:{event.action}"
        if self.whitelist.check(wl_key):
            return DecisionResult(
                action=ResponseAction.SILENT_MONITOR,
                risk_score=risk,
                subject_consistency=s_u,
                behavior_deviation=d_b,
                env_confidence=c_e,
                is_whitelisted=True,
                reason="动态白名单放行",
            )

        # 尝试加入白名单
        if self.whitelist.should_whitelist(s_u, d_b):
            self.whitelist.add(wl_key)

        # Phase 5: 阈值决策
        if risk < self.theta_1:
            action = ResponseAction.SILENT_MONITOR
            reason = f"低风险 R={risk:.3f} < θ_1={self.theta_1:.3f}"
        elif risk < self.theta_2:
            action = ResponseAction.ALERT
            reason = f"中风险 θ_1={self.theta_1:.3f} ≤ R={risk:.3f} < θ_2={self.theta_2:.3f}"
        else:
            action = ResponseAction.BLOCK
            reason = f"高风险 R={risk:.3f} ≥ θ_2={self.theta_2:.3f}"

        return DecisionResult(
            action=action,
            risk_score=risk,
            subject_consistency=s_u,
            behavior_deviation=d_b,
            env_confidence=c_e,
            is_whitelisted=False,
            reason=reason,
        )

    @staticmethod
    def optimize_thresholds(
        risk_scores: np.ndarray,
        labels: np.ndarray,
        fpr_constraint: float = 0.05,
    ) -> Tuple[float, float]:
        """阈值优化（ROC 分析 + Youden's J）

        max_{θ_1, θ_2} Recall(θ_1, θ_2)  s.t.  FPR ≤ ε

        Args:
            risk_scores: 验证集风险评分
            labels: 真实标签（0=正常, 1=异常）
            fpr_constraint: FPR 约束 ε

        Returns:
            (θ_1, θ_2) 最优阈值对
        """
        from sklearn.metrics import roc_curve

        fpr, tpr, thresholds = roc_curve(labels, risk_scores)

        # Youden's J 指数
        j_scores = tpr - fpr
        valid_mask = fpr <= fpr_constraint
        if valid_mask.any():
            best_idx = np.argmax(j_scores * valid_mask)
        else:
            best_idx = np.argmax(j_scores)

        theta_2 = float(thresholds[best_idx])
        theta_1 = 0.6 * theta_2

        return theta_1, theta_2
