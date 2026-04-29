"""
三维特征空间构建

从对齐后的事件流中提取用户的三维特征向量：
- 维度1: 内容语义特征（768维，Sentence-BERT）
- 维度2: 行为模式特征（96维 = 时间分布48 + 操作序列32 + 资源访问16）
- 维度3: 权限上下文特征（~32维，LDAP/HR/IAM）

论文对应：
    - 公式 (eq:time-normalize): 时间分布归一化
    - 公式 (eq:time-entropy): 时间熵
    - 公式 (eq:trigram-prob): 三元组条件概率
    - 公式 (eq:ig-filter): 信息增益过滤
    - 公式 (eq:resource-zscore): Z-score 标准化
"""

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import entropy as scipy_entropy


class FeatureExtractor:
    """三维特征空间构建器

    Args:
        time_bins: 时间分布的分箱数（48 = 24h / 30min）
        op_seq_dim: 操作序列特征维度（IG 过滤后）
        resource_dim: 资源访问特征维度
    """

    # 操作类型映射
    ACTION_TYPES = [
        "READ", "WRITE", "DELETE", "COPY",
        "SEND", "UPLOAD", "DOWNLOAD", "PRINT",
    ]

    def __init__(
        self,
        time_bins: int = 48,
        op_seq_dim: int = 32,
        resource_dim: int = 16,
    ):
        self.time_bins = time_bins
        self.op_seq_dim = op_seq_dim
        self.resource_dim = resource_dim
        self._ig_selected_indices = None  # 信息增益筛选后的特征索引

    # ----------------------------------------------------------------
    # 维度1: 内容语义特征（768维）
    # ----------------------------------------------------------------

    @staticmethod
    def compute_content_semantic(
        doc_vectors: List[np.ndarray],
        access_frequencies: List[float],
        dwell_times: List[float],
    ) -> np.ndarray:
        """计算用户内容语义特征（加权平均）

        c_u = Σ(w_i · v_i) / Σ(w_i)
        其中 w_i = access_frequency_i × dwell_time_i

        Args:
            doc_vectors: 文档语义向量列表 [v_1, v_2, ..., v_n]，每个 768维
            access_frequencies: 文档访问频次
            dwell_times: 文档停留时长

        Returns:
            768维内容语义特征向量
        """
        if not doc_vectors:
            return np.zeros(768)

        weights = np.array(access_frequencies) * np.array(dwell_times)
        vectors = np.stack(doc_vectors)
        weighted_sum = np.sum(weights[:, np.newaxis] * vectors, axis=0)
        return weighted_sum / (np.sum(weights) + 1e-8)

    # ----------------------------------------------------------------
    # 维度2: 行为模式特征（96维）
    # ----------------------------------------------------------------

    def compute_time_distribution(self, timestamps: List[float]) -> np.ndarray:
        """计算时间分布特征（48维）

        将 24 小时划分为 48 个 30 分钟间隔，统计操作分布。
        归一化为概率分布：t̂_{u,i} = t_{u,i} / Σ_j t_{u,j}

        Args:
            timestamps: 事件时间戳列表

        Returns:
            48维归一化时间分布向量
        """
        import datetime
        bins = np.zeros(self.time_bins)

        for ts in timestamps:
            dt = datetime.datetime.fromtimestamp(ts)
            bin_idx = (dt.hour * 60 + dt.minute) // 30  # 30分钟间隔
            bin_idx = min(bin_idx, self.time_bins - 1)
            bins[bin_idx] += 1

        # 归一化为概率分布
        total = bins.sum()
        if total > 0:
            bins = bins / total

        return bins

    @staticmethod
    def compute_time_entropy(time_distribution: np.ndarray) -> float:
        """计算时间熵 H_u^(t)

        H_u^(t) = -Σ t̂_{u,i} · log(t̂_{u,i})

        低熵 → 操作时间集中（正常办公节律）
        高熵 → 操作时间分散（可疑，如非工作时间活动）

        Args:
            time_distribution: 归一化时间分布

        Returns:
            时间熵值
        """
        # 过滤零值避免 log(0)
        nonzero = time_distribution[time_distribution > 0]
        return -np.sum(nonzero * np.log(nonzero + 1e-12))

    def compute_operation_sequence(
        self, action_sequence: List[str], preliminary_labels: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """计算操作序列特征（32维，IG 过滤后）

        1. 构建三元组滑动窗口统计
        2. 计算条件概率 P(e3|e1,e2) = N(e1,e2,e3) / N(e1,e2)
        3. 信息增益过滤：保留 IG > mean + σ 的特征维度
        4. 降维至 32 维

        Args:
            action_sequence: 操作类型序列 [READ, WRITE, COPY, ...]
            preliminary_labels: 初始聚类标签（用于计算信息增益）

        Returns:
            32维操作序列特征向量
        """
        # 构建三元组
        trigrams = Counter()
        bigrams = Counter()
        for i in range(len(action_sequence) - 2):
            tri = (action_sequence[i], action_sequence[i + 1], action_sequence[i + 2])
            bi = (action_sequence[i], action_sequence[i + 1])
            trigrams[tri] += 1
            bigrams[bi] += 1

        # 计算条件概率
        all_trigram_keys = sorted(trigrams.keys())
        trigram_probs = {}
        for tri in all_trigram_keys:
            bi = (tri[0], tri[1])
            if bigrams[bi] > 0:
                trigram_probs[tri] = trigrams[tri] / bigrams[bi]

        # 构建特征向量
        feature_vec = np.array([trigram_probs.get(k, 0.0) for k in all_trigram_keys])

        # 信息增益过滤
        if len(feature_vec) > self.op_seq_dim:
            if self._ig_selected_indices is not None:
                # 使用已计算的 IG 索引
                selected = self._ig_selected_indices[:self.op_seq_dim]
                selected = [i for i in selected if i < len(feature_vec)]
                feature_vec = feature_vec[selected]
            else:
                # 默认取方差最大的 32 维
                if len(feature_vec) > self.op_seq_dim:
                    indices = np.argsort(np.abs(feature_vec))[-self.op_seq_dim:]
                    feature_vec = feature_vec[indices]

        # 填充或截断至目标维度
        result = np.zeros(self.op_seq_dim)
        result[:min(len(feature_vec), self.op_seq_dim)] = feature_vec[:self.op_seq_dim]
        return result

    def compute_resource_access(
        self, access_counts: Dict[str, int], global_stats: Dict[str, Tuple[float, float]] = None
    ) -> np.ndarray:
        """计算资源访问特征（16维，Z-score 标准化）

        r_{u,i} = (f_{u,i} - μ_i) / σ_i

        Args:
            access_counts: 各资产类别的访问计数 {category: count}
            global_stats: 全局统计 {category: (mean, std)}

        Returns:
            16维标准化资源访问特征
        """
        # 预定义 16 类资产类别
        categories = [
            "code_repo", "financial_doc", "hr_record", "technical_spec",
            "email_attachment", "cloud_storage", "usb_device", "im_file",
            "database", "config_file", "log_file", "backup",
            "encrypted", "watermarked", "scanned", "other",
        ]

        raw_counts = np.array([access_counts.get(cat, 0) for cat in categories],
                               dtype=np.float64)

        if global_stats:
            # Z-score 标准化
            means = np.array([global_stats.get(cat, (0, 1))[0] for cat in categories])
            stds = np.array([global_stats.get(cat, (0, 1))[1] for cat in categories])
            stds = np.maximum(stds, 1e-8)  # 避免除以零
            normalized = (raw_counts - means) / stds
        else:
            # 简单归一化
            total = raw_counts.sum()
            normalized = raw_counts / (total + 1e-8) if total > 0 else raw_counts

        return normalized

    def compute_behavior_feature(
        self,
        timestamps: List[float],
        action_sequence: List[str],
        access_counts: Dict[str, int],
        global_stats: Dict[str, Tuple[float, float]] = None,
    ) -> np.ndarray:
        """计算完整行为模式特征（96维）

        拼接：时间分布(48) + 操作序列(32) + 资源访问(16)

        Returns:
            96维行为模式特征向量
        """
        time_feat = self.compute_time_distribution(timestamps)  # 48d
        op_feat = self.compute_operation_sequence(action_sequence)  # 32d
        resource_feat = self.compute_resource_access(access_counts, global_stats)  # 16d
        return np.concatenate([time_feat, op_feat, resource_feat])

    # ----------------------------------------------------------------
    # 维度3: 权限上下文特征（~32维）
    # ----------------------------------------------------------------

    @staticmethod
    def compute_permission_context(
        role: str,
        department: str,
        projects: List[str],
        permissions: List[str],
        role_vocab: List[str],
        dept_vocab: List[str],
        project_vocab: List[str],
        perm_vocab: List[str],
    ) -> np.ndarray:
        """计算权限上下文特征（多热编码）

        将角色、部门、项目、权限信息编码为二值向量。

        Returns:
            权限上下文特征向量
        """
        # 角色编码
        role_vec = np.array([1.0 if r == role else 0.0 for r in role_vocab])
        # 部门编码
        dept_vec = np.array([1.0 if d == department else 0.0 for d in dept_vocab])
        # 项目编码（多热）
        proj_vec = np.array([1.0 if p in projects else 0.0 for p in project_vocab])
        # 权限编码（多热）
        perm_vec = np.array([1.0 if p in permissions else 0.0 for p in perm_vocab])

        return np.concatenate([role_vec, dept_vec, proj_vec, perm_vec])

    # ----------------------------------------------------------------
    # 信息增益计算（用于 IG 过滤和 IGW-Kmeans 初始化）
    # ----------------------------------------------------------------

    @staticmethod
    def compute_information_gain(
        feature_matrix: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """计算各特征维度的信息增益

        IG(j) = H(Y) - H(Y|X_j)
             = H(Y) - Σ_v P(X_j=v) · H(Y|X_j=v)

        Args:
            feature_matrix: 特征矩阵 X ∈ R^{n×d}
            labels: 聚类标签 Y ∈ {1,...,K}^n

        Returns:
            各维度的信息增益 IG ∈ R^d
        """
        n, d = feature_matrix.shape
        K = len(np.unique(labels))

        # H(Y) —— 标签熵
        label_counts = np.bincount(labels, minlength=K)
        label_probs = label_counts / n
        H_Y = scipy_entropy(label_probs + 1e-12)

        ig_values = np.zeros(d)

        for j in range(d):
            # 离散化特征（等频分箱）
            n_bins = min(10, len(np.unique(feature_matrix[:, j])))
            if n_bins <= 1:
                ig_values[j] = 0.0
                continue

            percentiles = np.linspace(0, 100, n_bins + 1)
            bin_edges = np.percentile(feature_matrix[:, j], percentiles)
            bin_edges = np.unique(bin_edges)
            digitized = np.digitize(feature_matrix[:, j], bin_edges[1:-1])

            # H(Y|X_j)
            H_Y_given_Xj = 0.0
            for v in np.unique(digitized):
                mask = digitized == v
                p_v = mask.sum() / n
                if p_v == 0:
                    continue
                label_counts_v = np.bincount(labels[mask], minlength=K)
                label_probs_v = label_counts_v / mask.sum()
                H_Y_given_Xj += p_v * scipy_entropy(label_probs_v + 1e-12)

            ig_values[j] = H_Y - H_Y_given_Xj

        return ig_values

    def select_features_by_ig(
        self, feature_matrix: np.ndarray, labels: np.ndarray, threshold: str = "mean_plus_sigma"
    ) -> np.ndarray:
        """基于信息增益进行特征选择

        筛选规则：IG(g) > IG_mean + σ_IG

        Returns:
            选中的特征索引
        """
        ig = self.compute_information_gain(feature_matrix, labels)

        if threshold == "mean_plus_sigma":
            cutoff = ig.mean() + ig.std()
        else:
            cutoff = ig.mean()

        selected = np.where(ig > cutoff)[0]
        self._ig_selected_indices = selected
        return selected

    # ----------------------------------------------------------------
    # 构建完整用户特征矩阵
    # ----------------------------------------------------------------

    def build_user_feature_matrix(
        self, user_features: List[Dict]
    ) -> np.ndarray:
        """构建完整用户特征矩阵

        将多用户的三维特征拼接为矩阵 X ∈ R^{n×d}

        Args:
            user_features: 用户特征字典列表，每个包含：
                - content_semantic: 768维
                - behavior_pattern: 96维
                - permission_context: ~32维

        Returns:
            特征矩阵 X ∈ R^{n×(768+96+32)}
        """
        feature_list = []
        for uf in user_features:
            content = uf.get("content_semantic", np.zeros(768))
            behavior = uf.get("behavior_pattern", np.zeros(96))
            permission = uf.get("permission_context", np.zeros(32))
            feature_list.append(np.concatenate([content, behavior, permission]))

        return np.stack(feature_list)
