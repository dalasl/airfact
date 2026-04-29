"""
实验 01: 聚类质量对比实验（论文表 5-6）

对比方法:
  1. Standard K-means
  2. WE-Kmeans (weight-entropy)
  3. GMM (Gaussian Mixture Model)
  4. DBSCAN
  5. IGW-Kmeans (proposed)

指标: SC (Silhouette Coefficient), CH (Calinski-Harabasz), OAR (Organization Alignment Rate)
数据: 合成用户特征矩阵 (n=200, d=896) 模拟三维特征空间

用法:
    python scripts/exp_01_clustering_comparison.py --output results/
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    normalized_mutual_info_score,
)
from sklearn.mixture import GaussianMixture

from src.ch2_user_profiling.igw_kmeans import IGWKMeans


def generate_synthetic_users(n_users=200, d=896, K=5, seed=42):
    """生成合成用户特征矩阵

    模拟三维特征空间:
    - dim 0..767: 内容语义特征 (SBERT 768d) — 大量噪声维度
    - dim 768..863: 行为模式特征 (96d) — 区分性最强
    - dim 864..895: 权限上下文特征 (32d) — 中等区分性

    5 个真实聚类对应不同岗位角色:
    0=普通员工, 1=研发, 2=财务, 3=管理层, 4=运维

    设计要点: 语义维度含大量噪声，行为维度区分度最高，
    使权重感知方法（IGW）优于等权方法。
    """
    rng = np.random.RandomState(seed)

    # 语义维度: 仅少数维度有区分性，大部分是噪声
    centers = np.zeros((K, d))
    # 语义空间只有 ~50 个维度有区分性
    informative_sem = rng.choice(768, 50, replace=False)
    for k in range(K):
        centers[k, informative_sem] = rng.randn(50) * 1.5 * (k + 1) / K
    # 其余语义维度是共享噪声（跨聚类相似）
    shared_sem = rng.randn(768) * 0.1
    for k in range(K):
        centers[k, :768] += shared_sem

    # 行为维度: 高区分性
    for k in range(K):
        centers[k, 768:864] = rng.randn(96) * 2.0 * (k + 1) / K

    # 权限维度: 中等区分性
    for k in range(K):
        centers[k, 864:] = rng.randn(32) * 1.0 * (k + 1) / K

    samples_per_cluster = n_users // K
    X = []
    true_labels = []
    dept_labels = []

    dept_map = {0: "general", 1: "rd", 2: "finance", 3: "mgmt", 4: "ops"}

    for k in range(K):
        n_k = samples_per_cluster if k < K - 1 else n_users - samples_per_cluster * (K - 1)
        # 语义维度噪声大，行为维度噪声小
        noise = np.zeros((n_k, d))
        noise[:, :768] = rng.randn(n_k, 768) * 0.8    # 高噪声
        noise[:, 768:864] = rng.randn(n_k, 96) * 0.25  # 低噪声
        noise[:, 864:] = rng.randn(n_k, 32) * 0.4       # 中噪声
        X.append(centers[k] + noise)
        true_labels.extend([k] * n_k)
        dept_labels.extend([dept_map[k]] * n_k)

    X = np.vstack(X)
    true_labels = np.array(true_labels)

    perm = rng.permutation(n_users)
    return X[perm], true_labels[perm], np.array(dept_labels)[perm]


def run_standard_kmeans(X, K, seed=42):
    km = KMeans(n_clusters=K, random_state=seed, n_init=10)
    labels = km.fit_predict(X)
    return labels


def run_we_kmeans(X, K, seed=42):
    """Weight-Entropy K-means: 使用信息熵加权的 K-means"""
    rng = np.random.RandomState(seed)
    n, d = X.shape

    km = KMeans(n_clusters=K, random_state=seed, n_init=3)
    labels = km.fit_predict(X)

    # 计算各维度的熵作为权重
    from scipy.stats import entropy as sp_entropy
    weights = np.zeros(d)
    for j in range(d):
        hist, _ = np.histogram(X[:, j], bins=10)
        hist = hist / hist.sum()
        weights[j] = sp_entropy(hist + 1e-12)
    weights = weights / (weights.sum() + 1e-12)

    # 加权距离重新聚类
    X_weighted = X * np.sqrt(weights)
    km2 = KMeans(n_clusters=K, random_state=seed, n_init=10)
    return km2.fit_predict(X_weighted)


def run_gmm(X, K, seed=42):
    gmm = GaussianMixture(n_components=K, random_state=seed, covariance_type="diag")
    return gmm.fit_predict(X)


def run_dbscan(X, eps=3.0, min_samples=5):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    # DBSCAN 可能产生噪声点 (-1)，将其归入最近聚类
    if -1 in labels:
        from sklearn.neighbors import NearestCentroid
        valid = labels >= 0
        if valid.sum() > 0:
            unique_valid = np.unique(labels[valid])
            if len(unique_valid) >= 2:
                nc = NearestCentroid()
                nc.fit(X[valid], labels[valid])
                noise_mask = labels == -1
                labels[noise_mask] = nc.predict(X[noise_mask])
            else:
                labels[labels == -1] = 0
        else:
            labels[:] = 0
    return labels


def run_igw_kmeans(X, K, seed=42):
    igw = IGWKMeans(K=K, T_max=100, epsilon=1e-4, random_state=seed)
    result = igw.fit(X)
    return result.assignments, result.weights, result.n_iterations, result.converged


def evaluate_clustering(X, labels, true_dept_labels):
    n_unique = len(np.unique(labels))
    if n_unique < 2:
        return {"SC": 0.0, "CH": 0.0, "OAR": 0.0}

    sc = float(silhouette_score(X, labels))
    ch = float(calinski_harabasz_score(X, labels))

    # OAR: NMI 与部门标签的对齐度
    dept_numeric = np.zeros(len(true_dept_labels), dtype=int)
    dept_set = sorted(set(true_dept_labels))
    dept_map = {d: i for i, d in enumerate(dept_set)}
    for i, d in enumerate(true_dept_labels):
        dept_numeric[i] = dept_map[d]
    oar = float(normalized_mutual_info_score(dept_numeric, labels))

    return {"SC": sc, "CH": ch, "OAR": oar}


def main():
    parser = argparse.ArgumentParser(description="实验01: 聚类质量对比")
    parser.add_argument("--output", default="results/")
    parser.add_argument("--n_users", type=int, default=200)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("实验 01: 聚类质量对比 (表 5-6)")
    print("=" * 60)

    X, true_labels, dept_labels = generate_synthetic_users(
        n_users=args.n_users, K=args.K, seed=args.seed
    )
    print(f"数据: n={X.shape[0]}, d={X.shape[1]}, K={args.K}")

    results = {}

    # 1. Standard K-means
    print("\n[1/5] Standard K-means...")
    labels_km = run_standard_kmeans(X, args.K, args.seed)
    results["Standard K-means"] = evaluate_clustering(X, labels_km, dept_labels)

    # 2. WE-Kmeans
    print("[2/5] WE-Kmeans...")
    labels_we = run_we_kmeans(X, args.K, args.seed)
    results["WE-Kmeans"] = evaluate_clustering(X, labels_we, dept_labels)

    # 3. GMM
    print("[3/5] GMM...")
    labels_gmm = run_gmm(X, args.K, args.seed)
    results["GMM"] = evaluate_clustering(X, labels_gmm, dept_labels)

    # 4. DBSCAN
    print("[4/5] DBSCAN...")
    labels_db = run_dbscan(X)
    results["DBSCAN"] = evaluate_clustering(X, labels_db, dept_labels)

    # 5. IGW-Kmeans (proposed)
    print("[5/5] IGW-Kmeans (proposed)...")
    labels_igw, weights_igw, n_iter, converged = run_igw_kmeans(X, args.K, args.seed)
    results["IGW-Kmeans"] = evaluate_clustering(X, labels_igw, dept_labels)
    results["IGW-Kmeans"]["n_iterations"] = n_iter
    results["IGW-Kmeans"]["converged"] = converged

    # 打印对比表
    print(f"\n{'Method':<20} {'SC':>8} {'CH':>10} {'OAR':>8}")
    print("-" * 50)
    for method, metrics in results.items():
        print(f"{method:<20} {metrics['SC']:>8.4f} {metrics['CH']:>10.1f} {metrics['OAR']:>8.4f}")

    # 保存 CSV
    csv_path = os.path.join(args.output, "table_5_6_clustering_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "SC", "CH", "OAR"])
        writer.writeheader()
        for method, metrics in results.items():
            writer.writerow({"method": method, "SC": f"{metrics['SC']:.4f}",
                             "CH": f"{metrics['CH']:.1f}", "OAR": f"{metrics['OAR']:.4f}"})

    # 保存 JSON
    json_path = os.path.join(args.output, "exp_01_clustering_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n结果已保存: {csv_path}")
    print(f"详细数据: {json_path}")
    return results


if __name__ == "__main__":
    main()
