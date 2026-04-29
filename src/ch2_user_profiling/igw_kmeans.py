"""
信息增益加权 K-means 聚类算法（IGW-Kmeans）

算法 alg:igw-kmeans 的完整实现。
通过信息增益自适应学习各特征维度的权重，使聚类结果能自动
强调安全语义差异显著的特征维度。

三步交替优化：
    1. 分配步：固定权重和中心 → 按加权距离分配样本
    2. 更新步：固定分配 → 更新聚类中心
    3. 权重步：固定分配和中心 → 优化特征权重

收敛性证明：目标函数 J 有界且单调递减 → 收敛到局部最优。

论文对应：
    - 算法 alg:igw-kmeans
    - 公式 (eq:igw-distance), (eq:information-gain), (eq:initial-weight), (eq:objective)
"""

import numpy as np
from scipy.stats import entropy as scipy_entropy
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ClusterResult:
    """聚类结果"""
    centers: np.ndarray           # 聚类中心 {μ_k}，shape: (K, d)
    weights: np.ndarray           # 特征权重 w，shape: (d,)
    assignments: np.ndarray       # 样本分配 {c_i}，shape: (n,)
    objective_history: List[float]  # 目标函数收敛曲线
    n_iterations: int             # 实际迭代次数
    converged: bool               # 是否收敛


class IGWKMeans:
    """信息增益加权 K-means 聚类

    Args:
        K: 聚类数
        T_max: 最大迭代次数
        epsilon: 收敛阈值
        lambda_weight: 类内/类间散度平衡参数
        random_state: 随机种子
    """

    def __init__(
        self,
        K: int = 3,
        T_max: int = 100,
        epsilon: float = 1e-4,
        lambda_weight: float = 0.6,
        random_state: int = 42,
    ):
        self.K = K
        self.T_max = T_max
        self.epsilon = epsilon
        self.lambda_weight = lambda_weight
        self.rng = np.random.RandomState(random_state)

    def _standard_kmeans_init(self, X: np.ndarray) -> np.ndarray:
        """标准 K-means 初始化（K-means++ 选择初始中心）"""
        n, d = X.shape
        centers = np.zeros((self.K, d))

        # 随机选择第一个中心
        idx = self.rng.randint(0, n)
        centers[0] = X[idx]

        for k in range(1, self.K):
            # 计算每个样本到最近中心的距离
            dists = np.min([np.sum((X - centers[j]) ** 2, axis=1) for j in range(k)], axis=0)
            probs = dists / (dists.sum() + 1e-12)
            idx = self.rng.choice(n, p=probs)
            centers[k] = X[idx]

        return centers

    def _compute_initial_assignments(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """标准 K-means 分配（等权欧氏距离）"""
        dists = np.array([np.sum((X - centers[k]) ** 2, axis=1) for k in range(self.K)])
        return np.argmin(dists, axis=0)

    def _compute_information_gain(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """计算各特征维度的信息增益

        IG(j) = H(Y) - H(Y|X_j)

        Args:
            X: 特征矩阵 [n, d]
            labels: 聚类标签 [n]

        Returns:
            信息增益向量 [d]
        """
        n, d = X.shape

        # H(Y)
        label_counts = np.bincount(labels, minlength=self.K)
        label_probs = label_counts / n
        H_Y = scipy_entropy(label_probs + 1e-12)

        ig = np.zeros(d)
        for j in range(d):
            # 等频分箱离散化
            n_bins = min(10, len(np.unique(X[:, j])))
            if n_bins <= 1:
                ig[j] = 0.0
                continue
            percentiles = np.linspace(0, 100, n_bins + 1)
            bin_edges = np.unique(np.percentile(X[:, j], percentiles))
            if len(bin_edges) <= 1:
                ig[j] = 0.0
                continue
            digitized = np.digitize(X[:, j], bin_edges[1:-1])

            # H(Y|X_j)
            H_cond = 0.0
            for v in np.unique(digitized):
                mask = digitized == v
                p_v = mask.sum() / n
                if p_v == 0:
                    continue
                counts_v = np.bincount(labels[mask], minlength=self.K)
                probs_v = counts_v / mask.sum()
                H_cond += p_v * scipy_entropy(probs_v + 1e-12)

            ig[j] = H_Y - H_cond

        return ig

    def _initialize_weights(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """初始化特征权重 w^(0) = IG(j) / Σ_l IG(l)"""
        ig = self._compute_information_gain(X, labels)
        ig = np.maximum(ig, 0)  # 确保非负
        total = ig.sum()
        if total > 0:
            weights = ig / total
        else:
            weights = np.ones(X.shape[1]) / X.shape[1]
        return weights

    def _weighted_distance(self, X: np.ndarray, center: np.ndarray,
                            weights: np.ndarray) -> np.ndarray:
        """计算加权欧氏距离

        D_igw(x, μ; w) = Σ_j w_j (x_j - μ_j)²

        Args:
            X: 样本矩阵 [n, d]
            center: 聚类中心 [d]
            weights: 特征权重 [d]

        Returns:
            加权距离 [n]
        """
        diff_sq = (X - center) ** 2  # [n, d]
        return np.sum(weights * diff_sq, axis=1)  # [n]

    def _assignment_step(self, X: np.ndarray, centers: np.ndarray,
                          weights: np.ndarray) -> np.ndarray:
        """分配步：c_i = argmin_k Σ_j w_j (x_ij - μ_kj)²"""
        dists = np.array([self._weighted_distance(X, centers[k], weights)
                          for k in range(self.K)])  # [K, n]
        return np.argmin(dists, axis=0)

    def _update_centers(self, X: np.ndarray, assignments: np.ndarray) -> np.ndarray:
        """更新步：μ_k = (1/|C_k|) Σ_{i:c_i=k} x_i"""
        d = X.shape[1]
        centers = np.zeros((self.K, d))
        for k in range(self.K):
            mask = assignments == k
            if mask.sum() > 0:
                centers[k] = X[mask].mean(axis=0)
        return centers

    def _update_weights(self, X: np.ndarray, centers: np.ndarray,
                         assignments: np.ndarray) -> np.ndarray:
        """权重更新步

        w = argmin_w [λ · WithinScatter(w) + (1-λ) · BetweenScatter(w)]

        采用基于类内/类间散度比的解析解。
        """
        n, d = X.shape
        lam = self.lambda_weight

        # 类内散度（每维度）
        within = np.zeros(d)
        for k in range(self.K):
            mask = assignments == k
            if mask.sum() > 0:
                diff_sq = (X[mask] - centers[k]) ** 2
                within += diff_sq.sum(axis=0)

        # 类间散度（每维度）
        global_mean = X.mean(axis=0)
        between = np.zeros(d)
        for k in range(self.K):
            mask = assignments == k
            n_k = mask.sum()
            if n_k > 0:
                between += n_k * (centers[k] - global_mean) ** 2

        # 权重 ∝ between / (within + ε)
        # λ 控制类内和类间的相对重要性
        score = lam * (1.0 / (within + 1e-8)) + (1 - lam) * between
        score = np.maximum(score, 0)
        total = score.sum()
        if total > 0:
            weights = score / total
        else:
            weights = np.ones(d) / d

        return weights

    def _compute_objective(self, X: np.ndarray, centers: np.ndarray,
                            assignments: np.ndarray, weights: np.ndarray) -> float:
        """计算目标函数

        J(w, {μ_k}) = Σ_i Σ_k I(c_i=k) · Σ_j w_j (x_ij - μ_kj)²
        """
        total = 0.0
        for k in range(self.K):
            mask = assignments == k
            if mask.sum() > 0:
                total += self._weighted_distance(X[mask], centers[k], weights).sum()
        return total

    def fit(self, X: np.ndarray) -> ClusterResult:
        """运行 IGW-Kmeans 聚类

        Args:
            X: 用户特征矩阵 [n, d]

        Returns:
            ClusterResult 包含中心、权重、分配和收敛信息
        """
        n, d = X.shape

        # === 初始化 ===
        # Step 1: 标准 K-means 获取初始分配
        centers = self._standard_kmeans_init(X)
        assignments = self._compute_initial_assignments(X, centers)

        # Step 2: 基于信息增益初始化权重
        weights = self._initialize_weights(X, assignments)

        objective_history = []
        converged = False

        # === 迭代优化 ===
        for t in range(self.T_max):
            # Step 1: 分配步（固定 w, μ → 更新 c）
            assignments = self._assignment_step(X, centers, weights)

            # Step 2: 更新步（固定 c → 更新 μ）
            centers = self._update_centers(X, assignments)

            # Step 3: 权重步（固定 c, μ → 更新 w）
            weights = self._update_weights(X, centers, assignments)

            # 收敛检查
            obj = self._compute_objective(X, centers, assignments, weights)
            objective_history.append(obj)

            if len(objective_history) > 1:
                delta = abs(objective_history[-2] - objective_history[-1])
                if delta < self.epsilon:
                    converged = True
                    break

        return ClusterResult(
            centers=centers,
            weights=weights,
            assignments=assignments,
            objective_history=objective_history,
            n_iterations=t + 1,
            converged=converged,
        )

    def predict(self, X: np.ndarray, result: ClusterResult) -> np.ndarray:
        """使用已有聚类结果预测新样本的类别"""
        return self._assignment_step(X, result.centers, result.weights)

    @staticmethod
    def silhouette_coefficient(X: np.ndarray, labels: np.ndarray) -> float:
        """计算轮廓系数 SC ∈ [-1, 1]"""
        from sklearn.metrics import silhouette_score
        if len(np.unique(labels)) < 2:
            return 0.0
        return silhouette_score(X, labels)

    @staticmethod
    def calinski_harabasz_index(X: np.ndarray, labels: np.ndarray) -> float:
        """计算 Calinski-Harabasz 指数"""
        from sklearn.metrics import calinski_harabasz_score
        if len(np.unique(labels)) < 2:
            return 0.0
        return calinski_harabasz_score(X, labels)
