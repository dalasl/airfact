"""
事件驱动增量更新机制

支持两种触发模式：
1. 显式事件触发：HR 岗位变更、项目切换、权限提升 → 全量重算
2. 隐式漂移触发：Mahalanobis 距离超过阈值 → 增量更新

论文对应：
    - 公式 (eq:mahalanobis): D = √((x̄_u - μ_c)ᵀ S_c⁻¹ (x̄_u - μ_c))
    - 公式 (eq:incremental-update): μ_k^new = (1-η)μ_k^old + η·mean(C_k∩B)
"""

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import mahalanobis


class UpdateTriggerType(Enum):
    """更新触发类型"""
    EXPLICIT_EVENT = "explicit"    # 显式事件（岗位变更等）
    IMPLICIT_DRIFT = "implicit"    # 隐式漂移（行为偏移）


@dataclass
class ProfileVersion:
    """画像版本对象

    v_t = {
        cluster_centers: {μ_k^(t)},
        feature_weights: w^(t),
        timestamp: t_s^(t)
    }
    """
    cluster_centers: np.ndarray    # shape: (K, d)
    feature_weights: np.ndarray    # shape: (d,)
    timestamp: float
    version: int = 0
    covariance_matrices: Optional[List[np.ndarray]] = None  # 各聚类的协方差矩阵


@dataclass
class DriftDetectionResult:
    """漂移检测结果"""
    is_drifted: bool
    distance: float               # Mahalanobis 距离
    threshold: float              # 阈值 θ_drift
    user_id: str = ""
    cluster_id: int = -1


class IncrementalUpdater:
    """事件驱动增量更新器

    Args:
        eta: 增量学习率，默认 0.15
        drift_window: 漂移检测窗口大小（最近 N 次操作），默认 30
        theta_drift: Mahalanobis 距离阈值，默认 2.5
        full_recompute_window_days: 全量重算数据窗口（天），默认 7
    """

    def __init__(
        self,
        eta: float = 0.15,
        drift_window: int = 30,
        theta_drift: float = 2.5,
        full_recompute_window_days: int = 7,
    ):
        self.eta = eta
        self.drift_window = drift_window
        self.theta_drift = theta_drift
        self.full_recompute_window_days = full_recompute_window_days

        # 用户操作历史（用于漂移检测）
        self._user_histories: Dict[str, deque] = {}

    def _get_user_history(self, user_id: str) -> deque:
        """获取用户操作历史队列"""
        if user_id not in self._user_histories:
            self._user_histories[user_id] = deque(maxlen=self.drift_window)
        return self._user_histories[user_id]

    def record_operation(self, user_id: str, feature_vector: np.ndarray):
        """记录用户操作特征向量"""
        history = self._get_user_history(user_id)
        history.append(feature_vector)

    def detect_drift(
        self,
        user_id: str,
        cluster_center: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> DriftDetectionResult:
        """隐式漂移检测

        计算 Mahalanobis 距离：
        D = √((x̄_u - μ_c)ᵀ S_c⁻¹ (x̄_u - μ_c))

        其中 x̄_u 为用户最近 30 次操作的平均特征向量。

        Args:
            user_id: 用户 ID
            cluster_center: 所属聚类中心 μ_c
            covariance_matrix: 聚类协方差矩阵 S_c

        Returns:
            漂移检测结果
        """
        history = self._get_user_history(user_id)

        if len(history) < self.drift_window // 2:
            return DriftDetectionResult(
                is_drifted=False, distance=0.0,
                threshold=self.theta_drift, user_id=user_id
            )

        # 计算最近操作的平均特征
        x_bar = np.mean(list(history), axis=0)

        # 正则化协方差矩阵（避免奇异）
        reg_cov = covariance_matrix + np.eye(covariance_matrix.shape[0]) * 1e-6

        try:
            cov_inv = np.linalg.inv(reg_cov)
            diff = x_bar - cluster_center
            dist = np.sqrt(diff @ cov_inv @ diff)
        except np.linalg.LinAlgError:
            dist = np.linalg.norm(x_bar - cluster_center)

        return DriftDetectionResult(
            is_drifted=dist > self.theta_drift,
            distance=dist,
            threshold=self.theta_drift,
            user_id=user_id,
        )

    def incremental_update(
        self,
        profile: ProfileVersion,
        drifted_samples: np.ndarray,
        drifted_cluster_id: int,
    ) -> ProfileVersion:
        """增量更新聚类中心

        μ_k^new = (1-η) μ_k^old + η · mean(C_k ∩ B)

        Args:
            profile: 当前画像版本
            drifted_samples: 漂移样本集合 B [n_drift, d]
            drifted_cluster_id: 漂移发生的聚类 ID

        Returns:
            更新后的画像版本
        """
        new_centers = profile.cluster_centers.copy()
        k = drifted_cluster_id

        # 计算漂移样本均值
        drift_mean = drifted_samples.mean(axis=0)

        # 指数移动平均更新
        new_centers[k] = (1 - self.eta) * new_centers[k] + self.eta * drift_mean

        # 更新协方差（如有）
        new_cov = None
        if profile.covariance_matrices is not None:
            new_cov = [c.copy() for c in profile.covariance_matrices]
            drift_cov = np.cov(drifted_samples.T) if drifted_samples.shape[0] > 1 \
                else np.eye(drifted_samples.shape[1]) * 0.01
            new_cov[k] = (1 - self.eta) * new_cov[k] + self.eta * drift_cov

        return ProfileVersion(
            cluster_centers=new_centers,
            feature_weights=profile.feature_weights.copy(),
            timestamp=time.time(),
            version=profile.version + 1,
            covariance_matrices=new_cov,
        )

    def full_recompute(
        self,
        feature_matrix: np.ndarray,
        igw_kmeans_fn: Callable,
    ) -> ProfileVersion:
        """显式事件触发的全量重算

        使用最近 7 天的数据完整运行 IGW-Kmeans。

        Args:
            feature_matrix: 完整特征矩阵（7天窗口内）
            igw_kmeans_fn: IGW-Kmeans 聚类函数

        Returns:
            全新的画像版本
        """
        result = igw_kmeans_fn(feature_matrix)

        # 计算各聚类协方差矩阵
        cov_matrices = []
        for k in range(result.centers.shape[0]):
            mask = result.assignments == k
            if mask.sum() > 1:
                cov = np.cov(feature_matrix[mask].T)
            else:
                cov = np.eye(feature_matrix.shape[1]) * 0.01
            cov_matrices.append(cov)

        return ProfileVersion(
            cluster_centers=result.centers,
            feature_weights=result.weights,
            timestamp=time.time(),
            version=0,  # 全量重算重置版本号
            covariance_matrices=cov_matrices,
        )

    def process_update(
        self,
        user_id: str,
        current_profile: ProfileVersion,
        user_cluster_id: int,
        trigger_type: UpdateTriggerType,
        feature_matrix: np.ndarray = None,
        igw_kmeans_fn: Callable = None,
    ) -> Tuple[ProfileVersion, Dict]:
        """统一更新入口

        Args:
            user_id: 用户 ID
            current_profile: 当前画像
            user_cluster_id: 用户所属聚类
            trigger_type: 触发类型
            feature_matrix: 特征矩阵（全量重算时需要）
            igw_kmeans_fn: IGW-Kmeans 函数（全量重算时需要）

        Returns:
            (更新后画像, 更新元数据)
        """
        metadata = {
            "user_id": user_id,
            "trigger_type": trigger_type.value,
            "timestamp": time.time(),
        }

        if trigger_type == UpdateTriggerType.EXPLICIT_EVENT:
            # 全量重算
            assert feature_matrix is not None and igw_kmeans_fn is not None
            new_profile = self.full_recompute(feature_matrix, igw_kmeans_fn)
            metadata["update_method"] = "full_recompute"
            metadata["data_window_days"] = self.full_recompute_window_days

        elif trigger_type == UpdateTriggerType.IMPLICIT_DRIFT:
            # 增量更新
            history = self._get_user_history(user_id)
            drifted_samples = np.array(list(history))
            new_profile = self.incremental_update(
                current_profile, drifted_samples, user_cluster_id
            )
            metadata["update_method"] = "incremental"
            metadata["drift_samples"] = len(drifted_samples)
            metadata["eta"] = self.eta

        else:
            raise ValueError(f"未知触发类型: {trigger_type}")

        return new_profile, metadata
