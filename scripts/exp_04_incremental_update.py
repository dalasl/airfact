"""
实验 04: 增量更新有效性实验（论文表 5-9）

验证事件驱动增量更新机制的有效性:
  1. 无增量更新 (Static) — 初始聚类后固定
  2. 定期全量重聚类 (Periodic) — 每 N 步全量重算
  3. IGW + 增量更新 (Proposed) — 漂移检测 + 增量学习

指标: SC变化趋势, 检测到漂移次数, 更新延迟(ms)

用法:
    python scripts/exp_04_incremental_update.py --output results/
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import silhouette_score

from src.ch2_user_profiling.igw_kmeans import IGWKMeans
from src.ch2_user_profiling.incremental_update import (
    IncrementalUpdater,
    ProfileVersion,
    UpdateTriggerType,
)


def generate_evolving_data(n=200, d=896, K=5, n_steps=10, drift_step=5, seed=42):
    """生成带概念漂移的时序用户数据

    在 drift_step 步骤后，部分用户行为发生偏移（模拟岗位变更）。
    """
    rng = np.random.RandomState(seed)
    centers = []
    for k in range(K):
        c = rng.randn(d) * 0.5
        c[768:864] *= 2.0 + k * 0.5
        centers.append(c)

    snapshots = []
    for step in range(n_steps):
        step_centers = [c.copy() for c in centers]

        # 在 drift_step 后引入漂移
        if step >= drift_step:
            drift_intensity = (step - drift_step + 1) * 0.2
            # 聚类 1 (研发) 的行为模式向聚类 2 (财务) 漂移
            step_centers[1][768:864] += drift_intensity * (centers[2][768:864] - centers[1][768:864])
            # 聚类 3 (管理层) 的语义特征发生变化
            step_centers[3][:768] += rng.randn(768) * drift_intensity * 0.1

        X = []
        for k in range(K):
            n_k = n // K
            noise = rng.randn(n_k, d) * 0.3
            X.append(step_centers[k] + noise)

        snapshots.append(np.vstack(X))

    return snapshots


def main():
    parser = argparse.ArgumentParser(description="实验04: 增量更新有效性")
    parser.add_argument("--output", default="results/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("实验 04: 增量更新有效性 (表 5-9)")
    print("=" * 60)

    K = 5
    n_steps = 10
    drift_step = 5
    snapshots = generate_evolving_data(n=200, K=K, n_steps=n_steps,
                                       drift_step=drift_step, seed=args.seed)

    # 初始聚类
    igw = IGWKMeans(K=K, T_max=100, random_state=args.seed)
    init_result = igw.fit(snapshots[0])

    results = {"Static": [], "Periodic": [], "Proposed": []}

    # 方法 1: Static — 用初始模型分配所有后续数据
    for step, X in enumerate(snapshots):
        labels = igw.predict(X, init_result)
        sc = float(silhouette_score(X, labels)) if len(np.unique(labels)) >= 2 else 0.0
        results["Static"].append({"step": step, "SC": sc, "update_ms": 0})

    # 方法 2: Periodic — 每 3 步全量重聚类
    periodic_result = init_result
    for step, X in enumerate(snapshots):
        t0 = time.perf_counter()
        if step > 0 and step % 3 == 0:
            igw_p = IGWKMeans(K=K, T_max=100, random_state=args.seed + step)
            periodic_result = igw_p.fit(X)
        labels = igw.predict(X, periodic_result)
        update_ms = (time.perf_counter() - t0) * 1000
        sc = float(silhouette_score(X, labels)) if len(np.unique(labels)) >= 2 else 0.0
        results["Periodic"].append({"step": step, "SC": sc, "update_ms": update_ms})

    # 方法 3: Proposed — 漂移检测 + 增量更新
    updater = IncrementalUpdater(
        eta=0.15, theta_drift=3.0, drift_window=30,
    )

    # 创建初始 ProfileVersion
    cov_matrices = []
    for k in range(K):
        mask = init_result.assignments == k
        if mask.sum() > 1:
            cov = np.cov(snapshots[0][mask], rowvar=False) + np.eye(snapshots[0].shape[1]) * 1e-6
        else:
            cov = np.eye(snapshots[0].shape[1])
        cov_matrices.append(cov)

    current_version = ProfileVersion(
        cluster_centers=init_result.centers.copy(),
        feature_weights=init_result.weights.copy(),
        timestamp=time.time(),
        version=0,
        covariance_matrices=cov_matrices,
    )

    # 预填充用户操作历史（用初始数据）
    n_k = snapshots[0].shape[0] // K
    for k in range(K):
        user_id = f"cluster_{k}"
        for row in snapshots[0][k * n_k:(k + 1) * n_k]:
            updater.record_operation(user_id, row)

    drift_count = 0
    for step, X in enumerate(snapshots):
        t0 = time.perf_counter()

        # 记录当前步操作并检测漂移
        drifted = False
        drifted_k = -1
        for k in range(K):
            user_id = f"cluster_{k}"
            cluster_data = X[k * n_k:(k + 1) * n_k]
            for row in cluster_data:
                updater.record_operation(user_id, row)

            drift_result = updater.detect_drift(
                user_id=user_id,
                cluster_center=current_version.cluster_centers[k],
                covariance_matrix=current_version.covariance_matrices[k],
            )
            if drift_result.is_drifted:
                drifted = True
                drifted_k = k
                break

        if drifted:
            drift_count += 1
            drifted_samples = X[drifted_k * n_k:(drifted_k + 1) * n_k]
            current_version = updater.incremental_update(
                profile=current_version,
                drifted_samples=drifted_samples,
                drifted_cluster_id=drifted_k,
            )

        # 使用当前版本分配
        igw_cur = IGWKMeans(K=K)
        from src.ch2_user_profiling.igw_kmeans import ClusterResult
        cur_result = ClusterResult(
            centers=current_version.cluster_centers,
            weights=current_version.feature_weights,
            assignments=np.zeros(X.shape[0], dtype=int),
            objective_history=[], n_iterations=0, converged=True,
        )
        labels = igw_cur.predict(X, cur_result)
        update_ms = (time.perf_counter() - t0) * 1000
        sc = float(silhouette_score(X, labels)) if len(np.unique(labels)) >= 2 else 0.0
        results["Proposed"].append({"step": step, "SC": sc, "update_ms": update_ms})

    # 打印结果
    print(f"\n{'Step':>5}", end="")
    for method in ["Static", "Periodic", "Proposed"]:
        print(f"  {method+' SC':>12}", end="")
    print()
    print("-" * 50)

    for step in range(n_steps):
        print(f"{step:>5}", end="")
        for method in ["Static", "Periodic", "Proposed"]:
            sc = results[method][step]["SC"]
            print(f"  {sc:>12.4f}", end="")
        print("  *" if step >= drift_step else "")

    # 汇总
    print(f"\n漂移检测次数 (Proposed): {drift_count}")
    for method in ["Static", "Periodic", "Proposed"]:
        avg_sc = np.mean([r["SC"] for r in results[method]])
        post_drift_sc = np.mean([r["SC"] for r in results[method][drift_step:]])
        avg_ms = np.mean([r["update_ms"] for r in results[method]])
        print(f"  {method}: avg_SC={avg_sc:.4f}, post_drift_SC={post_drift_sc:.4f}, avg_update={avg_ms:.1f}ms")

    # 保存 CSV
    csv_path = os.path.join(args.output, "table_5_9_incremental_update.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "Static_SC", "Periodic_SC", "Proposed_SC"])
        writer.writeheader()
        for step in range(n_steps):
            writer.writerow({
                "step": step,
                "Static_SC": f"{results['Static'][step]['SC']:.4f}",
                "Periodic_SC": f"{results['Periodic'][step]['SC']:.4f}",
                "Proposed_SC": f"{results['Proposed'][step]['SC']:.4f}",
            })

    json_path = os.path.join(args.output, "exp_04_incremental_update.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n结果已保存: {csv_path}")
    return results


if __name__ == "__main__":
    main()
