"""
实验 03: K 值敏感性分析实验（论文表 5-8）

研究聚类数 K 对 IGW-Kmeans 性能的影响:
  K = 2, 3, 4, 5, 6, 7, 8

指标: SC, CH, 收敛迭代次数, 各聚类样本数

用法:
    python scripts/exp_03_k_sensitivity.py --output results/
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import silhouette_score, calinski_harabasz_score

from src.ch2_user_profiling.igw_kmeans import IGWKMeans


def generate_data(n=200, d=896, K_true=5, seed=42):
    rng = np.random.RandomState(seed)
    centers = []
    for k in range(K_true):
        c = rng.randn(d) * 0.5
        c[768:864] *= 2.0 + k * 0.5
        centers.append(c)

    X, labels = [], []
    for k in range(K_true):
        n_k = n // K_true if k < K_true - 1 else n - (n // K_true) * (K_true - 1)
        X.append(centers[k] + rng.randn(n_k, d) * 0.3)
        labels.extend([k] * n_k)

    X = np.vstack(X)
    perm = rng.permutation(n)
    return X[perm], np.array(labels)[perm]


def main():
    parser = argparse.ArgumentParser(description="实验03: K值敏感性分析")
    parser.add_argument("--output", default="results/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("实验 03: K 值敏感性分析 (表 5-8)")
    print("=" * 60)

    K_true = 5
    X, true_labels = generate_data(n=200, K_true=K_true, seed=args.seed)
    print(f"数据: n={X.shape[0]}, d={X.shape[1]}, K_true={K_true}")

    K_values = [2, 3, 4, 5, 6, 7, 8]
    results = {}

    print(f"\n{'K':>4} {'SC':>8} {'CH':>10} {'Iters':>6} {'Conv':>5} {'Cluster Sizes'}")
    print("-" * 70)

    for K in K_values:
        igw = IGWKMeans(K=K, T_max=100, epsilon=1e-4, random_state=args.seed)
        result = igw.fit(X)

        labels = result.assignments
        n_unique = len(np.unique(labels))

        sc = float(silhouette_score(X, labels)) if n_unique >= 2 else 0.0
        ch = float(calinski_harabasz_score(X, labels)) if n_unique >= 2 else 0.0

        # 聚类大小分布
        sizes = [int(np.sum(labels == k)) for k in range(K)]

        results[K] = {
            "SC": sc,
            "CH": ch,
            "n_iterations": result.n_iterations,
            "converged": result.converged,
            "cluster_sizes": sizes,
            "objective_final": float(result.objective_history[-1]) if result.objective_history else 0.0,
        }

        sizes_str = ", ".join(str(s) for s in sizes)
        conv_str = "Y" if result.converged else "N"
        print(f"{K:>4} {sc:>8.4f} {ch:>10.1f} {result.n_iterations:>6} {conv_str:>5} [{sizes_str}]")

    # 最优 K (最高 SC)
    best_K = max(results, key=lambda k: results[k]["SC"])
    print(f"\n最优 K = {best_K} (SC = {results[best_K]['SC']:.4f})")

    # 保存 CSV
    csv_path = os.path.join(args.output, "table_5_8_k_sensitivity.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["K", "SC", "CH", "n_iterations", "converged"])
        writer.writeheader()
        for K, m in results.items():
            writer.writerow({
                "K": K, "SC": f"{m['SC']:.4f}", "CH": f"{m['CH']:.1f}",
                "n_iterations": m["n_iterations"], "converged": m["converged"],
            })

    json_path = os.path.join(args.output, "exp_03_k_sensitivity.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in results.items()}, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {csv_path}")
    return results


if __name__ == "__main__":
    main()
