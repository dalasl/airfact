"""
评估指标计算

包含论文中使用的所有评估指标：
- 聚类质量：轮廓系数 (SC)、Calinski-Harabasz 指数 (CH)
- 分级性能：Accuracy、Precision、Recall、F1、FPR
- 规则质量：语法合规率、逻辑冲突率、语义保真度、执行成功率
- 统计检验：McNemar 检验、Wilcoxon 符号秩检验、Fleiss' Kappa
"""

import numpy as np
from typing import Dict, List, Tuple


class MetricsCalculator:
    """评估指标计算器"""

    # ============= 聚类质量指标 =============

    @staticmethod
    def silhouette_coefficient(X: np.ndarray, labels: np.ndarray) -> float:
        """轮廓系数 SC ∈ [-1, 1]"""
        from sklearn.metrics import silhouette_score
        if len(np.unique(labels)) < 2:
            return 0.0
        return float(silhouette_score(X, labels))

    @staticmethod
    def calinski_harabasz_index(X: np.ndarray, labels: np.ndarray) -> float:
        """Calinski-Harabasz 指数"""
        from sklearn.metrics import calinski_harabasz_score
        if len(np.unique(labels)) < 2:
            return 0.0
        return float(calinski_harabasz_score(X, labels))

    @staticmethod
    def organization_alignment_rate(
        predicted_labels: np.ndarray, department_labels: np.ndarray
    ) -> float:
        """组织对齐率（CERT 实验用）"""
        from sklearn.metrics import normalized_mutual_info_score
        return float(normalized_mutual_info_score(predicted_labels, department_labels))

    # ============= 分级性能指标 =============

    @staticmethod
    def classification_report(
        y_true: np.ndarray, y_pred: np.ndarray, labels: List[str] = None
    ) -> Dict:
        """分级综合报告"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score
        )
        labels = labels or ["L1", "L2", "L3", "L4"]

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "per_class": {
                label: {
                    "precision": float(precision_score(y_true == label, y_pred == label, zero_division=0)),
                    "recall": float(recall_score(y_true == label, y_pred == label, zero_division=0)),
                    "f1": float(f1_score(y_true == label, y_pred == label, zero_division=0)),
                }
                for label in labels
            },
        }

    @staticmethod
    def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """误报率 FPR"""
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        return float(fp / (fp + tn + 1e-8))

    @staticmethod
    def detection_metrics(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """检测指标（Recall + FPR + F1）"""
        from sklearn.metrics import recall_score, f1_score
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        recall = tp / (tp + fn + 1e-8)
        fpr = fp / (fp + tn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            "recall": float(recall),
            "fpr": float(fpr),
            "precision": float(precision),
            "f1": float(f1),
        }

    # ============= 规则质量指标 =============

    @staticmethod
    def syntax_compliance_rate(results: List[bool]) -> float:
        """语法合规率"""
        return sum(results) / len(results) if results else 0.0

    @staticmethod
    def logic_conflict_rate(results: List[bool]) -> float:
        """逻辑冲突率"""
        return sum(results) / len(results) if results else 0.0

    @staticmethod
    def execution_success_rate(results: List[bool]) -> float:
        """执行成功率"""
        return sum(results) / len(results) if results else 0.0

    # ============= 统计检验 =============

    @staticmethod
    def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray,
                      y_pred_b: np.ndarray) -> Tuple[float, float]:
        """McNemar 检验（分级方法显著性比较）"""
        from scipy.stats import chi2

        correct_a = (y_pred_a == y_true)
        correct_b = (y_pred_b == y_true)

        # 列联表
        b = np.sum(correct_a & ~correct_b)  # A对B错
        c = np.sum(~correct_a & correct_b)  # A错B对

        if b + c == 0:
            return 0.0, 1.0

        statistic = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - chi2.cdf(statistic, df=1)
        return float(statistic), float(p_value)

    @staticmethod
    def wilcoxon_signed_rank(scores_a: np.ndarray, scores_b: np.ndarray) -> Tuple[float, float]:
        """Wilcoxon 符号秩检验（聚类方法显著性比较）"""
        from scipy.stats import wilcoxon
        stat, p_value = wilcoxon(scores_a, scores_b)
        return float(stat), float(p_value)

    @staticmethod
    def fleiss_kappa(ratings: np.ndarray) -> float:
        """Fleiss' Kappa（标注一致性）"""
        n, k = ratings.shape
        N = ratings.sum(axis=1)[0]

        p_j = ratings.sum(axis=0) / (n * N)
        P_i = (np.sum(ratings ** 2, axis=1) - N) / (N * (N - 1))
        P_bar = P_i.mean()
        P_e = np.sum(p_j ** 2)

        kappa = (P_bar - P_e) / (1 - P_e + 1e-8)
        return float(kappa)

    # ============= 交叉验证工具 =============

    @staticmethod
    def stratified_kfold_evaluate(
        X: np.ndarray, y: np.ndarray, model_fn, n_folds: int = 5
    ) -> Dict[str, Tuple[float, float]]:
        """分层 K 折交叉验证"""
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        metrics_per_fold = []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = model_fn()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            fold_metrics = MetricsCalculator.classification_report(y_test, y_pred)
            metrics_per_fold.append(fold_metrics)

        # 聚合：均值 ± 标准差
        result = {}
        for key in ["accuracy", "precision", "recall", "f1"]:
            values = [m[key] for m in metrics_per_fold]
            result[key] = (float(np.mean(values)), float(np.std(values)))

        return result
