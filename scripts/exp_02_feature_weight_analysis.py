"""
实验 02: 信息增益权重分析实验（论文表 5-7）

对比不同权重策略对聚类质量的影响:
  1. 等权 (Equal)
  2. TF-IDF 权重
  3. PCA 方差贡献率
  4. IGW 信息增益权重 (proposed)

分析三维特征空间各维度的权重分布。

用法:
    python scripts/exp_02_feature_weight_analysis.py --output results/
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from src.ch2_user_profiling.igw_kmeans import IGWKMeans


def generate_data(n=200, d=896, K=5, seed=42):
    rng = np.random.RandomState(seed)
    centers = []
    for k in range(K):
        c = rng.randn(d) * 0.5
        c[768:864] *= 2.0 + k * 0.5  # 行为维度差异更大
        c[864:] *= 1.5 + k * 0.3     # 权限维度也有差异
        centers.append(c)

    X, labels = [], []
    for k in range(K):
        n_k = n // K if k < K - 1 else n - (n // K) * (K - 1)
        X.append(centers[k] + rng.randn(n_k, d) * 0.3)
        labels.extend([k] * n_k)

    X = np.vstack(X)
    labels = np.array(labels)
    perm = rng.permutation(n)
    return X[perm], labels[perm]


def equal_weights(d):
    return np.ones(d) / d


def tfidf_weights(X):
    """模拟 TF-IDF 权重: 基于特征频率的逆文档频率加权"""
    d = X.shape[1]
    # TF: 特征值归一化后的均值
    tf = np.abs(X).mean(axis=0)
    tf = tf / (tf.max() + 1e-12)
    # IDF: 非零频率的逆
    doc_freq = (np.abs(X) > 0.1).sum(axis=0) / X.shape[0]
    idf = np.log(1.0 / (doc_freq + 1e-12) + 1)
    w = tf * idf
    return w / (w.sum() + 1e-12)


def pca_weights(X):
    """PCA 方差贡献率作为权重"""
    d = X.shape[1]
    n_components = min(50, d, X.shape[0])
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X)

    # 将 component 方差映射回原始维度
    w = np.zeros(d)
    for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_ratio_)):
        w += var * np.abs(comp)
    w = np.maximum(w, 0)
    return w / (w.sum() + 1e-12)


def weighted_kmeans(X, weights, K, seed=42):
    X_w = X * np.sqrt(weights)
    km = KMeans(n_clusters=K, random_state=seed, n_init=10)
    labels = km.fit_predict(X_w)
    return labels


def main():
    parser = argparse.ArgumentParser(description="实验02: 信息增益权重分析")
    parser.add_argument("--output", default="results/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("实验 02: 信息增益权重有效性分析 (表 5-7)")
    print("=" * 60)

    K = 5
    X, true_labels = generate_data(n=200, K=K, seed=args.seed)
    d = X.shape[1]
    print(f"数据: n={X.shape[0]}, d={d}, K={K}")

    strategies = {}

    # 1. Equal
    w_eq = equal_weights(d)
    labels_eq = weighted_kmeans(X, w_eq, K, args.seed)
    strategies["Equal"] = {"weights": w_eq, "labels": labels_eq}

    # 2. TF-IDF
    w_tfidf = tfidf_weights(X)
    labels_tfidf = weighted_kmeans(X, w_tfidf, K, args.seed)
    strategies["TF-IDF"] = {"weights": w_tfidf, "labels": labels_tfidf}

    # 3. PCA
    w_pca = pca_weights(X)
    labels_pca = weighted_kmeans(X, w_pca, K, args.seed)
    strategies["PCA-Var"] = {"weights": w_pca, "labels": labels_pca}

    # 4. IGW (proposed)
    igw = IGWKMeans(K=K, T_max=100, random_state=args.seed)
    result_igw = igw.fit(X)
    strategies["IGW"] = {"weights": result_igw.weights, "labels": result_igw.assignments}

    # 评估
    results = {}
    print(f"\n{'Strategy':<12} {'SC':>8} {'CH':>10} {'Sem-W':>8} {'Beh-W':>8} {'Ctx-W':>8}")
    print("-" * 60)
    for name, info in strategies.items():
        labels = info["labels"]
        w = info["weights"]

        n_unique = len(np.unique(labels))
        sc = float(silhouette_score(X, labels)) if n_unique >= 2 else 0.0
        ch = float(calinski_harabasz_score(X, labels)) if n_unique >= 2 else 0.0

        # 三维权重分布
        sem_w = float(w[:768].sum())   # 语义维度权重和
        beh_w = float(w[768:864].sum())  # 行为维度权重和
        ctx_w = float(w[864:].sum())     # 上下文维度权重和

        results[name] = {
            "SC": sc, "CH": ch,
            "semantic_weight": sem_w, "behavior_weight": beh_w, "context_weight": ctx_w,
        }
        print(f"{name:<12} {sc:>8.4f} {ch:>10.1f} {sem_w:>8.3f} {beh_w:>8.3f} {ctx_w:>8.3f}")

    # 保存 CSV
    csv_path = os.path.join(args.output, "table_5_7_weight_analysis.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "strategy", "SC", "CH", "semantic_weight", "behavior_weight", "context_weight"
        ])
        writer.writeheader()
        for name, m in results.items():
            writer.writerow({"strategy": name, **{k: f"{v:.4f}" for k, v in m.items()}})

    json_path = os.path.join(args.output, "exp_02_feature_weight_analysis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {csv_path}")
    return results


if __name__ == "__main__":
    main()
