#!/usr/bin/env python3
"""
一键运行全部实验

按论文第5章顺序运行所有 11 个实验脚本，收集结果到 results/。

实验清单:
  exp_01: 聚类质量对比 (表5-6)
  exp_02: 信息增益权重分析 (表5-7)
  exp_03: K值敏感性分析 (表5-8)
  exp_04: 增量更新有效性 (表5-9)
  exp_05: 分级多基线对比 (表5-10)
  exp_06: 分格式分级性能 (表5-11)
  exp_07: 上下文增强消融 (表5-12)
  exp_08: 自洽性采样数影响 (表5-13)
  exp_09: 参数敏感性分析 (表5-14, 5-15)
  exp_10: 规则生成质量 (表5-16)
  exp_11: 端到端检测对比 (表5-4, 5-5)

用法:
    python scripts/run_all_experiments.py                  # 运行全部
    python scripts/run_all_experiments.py --stage profiling # 仅运行画像实验
    python scripts/run_all_experiments.py --exp 1 5 11      # 运行指定实验
"""

import argparse
import importlib
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXPERIMENTS = {
    1: ("exp_01_clustering_comparison", "聚类质量对比 (表5-6)"),
    2: ("exp_02_feature_weight_analysis", "信息增益权重分析 (表5-7)"),
    3: ("exp_03_k_sensitivity", "K值敏感性分析 (表5-8)"),
    4: ("exp_04_incremental_update", "增量更新有效性 (表5-9)"),
    5: ("exp_05_grading_baseline_comparison", "分级多基线对比 (表5-10)"),
    6: ("exp_06_format_grading", "分格式分级性能 (表5-11)"),
    7: ("exp_07_context_ablation", "上下文增强消融 (表5-12)"),
    8: ("exp_08_consistency_sampling", "自洽性采样数影响 (表5-13)"),
    9: ("exp_09_parameter_sensitivity", "参数敏感性分析 (表5-14,15)"),
    10: ("exp_10_rule_generation_quality", "规则生成质量 (表5-16)"),
    11: ("exp_11_e2e_detection", "端到端检测对比 (表5-4,5)"),
}

STAGE_MAP = {
    "profiling": [1, 2, 3, 4],
    "grading": [5, 6, 7, 8],
    "rule": [10],
    "e2e": [11],
    "sensitivity": [9],
}


def run_experiment(exp_id, output_dir):
    """运行单个实验脚本"""
    module_name, description = EXPERIMENTS[exp_id]
    print(f"\n{'=' * 60}")
    print(f"[{exp_id:02d}/11] {description}")
    print(f"{'=' * 60}")

    script_path = os.path.join(os.path.dirname(__file__), f"{module_name}.py")
    if not os.path.exists(script_path):
        print(f"  SKIP: {script_path} not found")
        return False, 0

    t0 = time.time()
    try:
        # 使用 importlib 动态导入并运行
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        mod = importlib.util.module_from_spec(spec)

        # 设置 sys.argv 以传递 --output 参数
        old_argv = sys.argv
        sys.argv = [script_path, "--output", output_dir]
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.argv = old_argv

        elapsed = time.time() - t0
        print(f"\n  PASS ({elapsed:.1f}s)")
        return True, elapsed

    except SystemExit:
        elapsed = time.time() - t0
        print(f"\n  PASS ({elapsed:.1f}s)")
        return True, elapsed

    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  FAIL: {e}")
        traceback.print_exc()
        return False, elapsed


def main():
    parser = argparse.ArgumentParser(description="运行全部实验")
    parser.add_argument("--output", type=str, default="results/",
                        help="输出目录")
    parser.add_argument("--stage", type=str, default=None,
                        choices=list(STAGE_MAP.keys()),
                        help="运行指定阶段")
    parser.add_argument("--exp", type=int, nargs="+", default=None,
                        help="运行指定实验编号 (1-11)")
    parser.add_argument("--seed", type=int, default=42,
                        help="全局随机种子")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 确定要运行的实验
    if args.exp:
        exp_ids = sorted(set(args.exp))
    elif args.stage:
        exp_ids = STAGE_MAP[args.stage]
    else:
        exp_ids = list(EXPERIMENTS.keys())

    # 验证实验编号
    invalid = [e for e in exp_ids if e not in EXPERIMENTS]
    if invalid:
        print(f"Invalid experiment IDs: {invalid}")
        print(f"Valid range: 1-{len(EXPERIMENTS)}")
        sys.exit(1)

    print(f"Experiment start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output: {args.output}")
    print(f"Experiments: {exp_ids}")

    start_time = time.time()
    results = {}

    for exp_id in exp_ids:
        success, elapsed = run_experiment(exp_id, args.output)
        results[exp_id] = {"success": success, "elapsed": elapsed}

    # 汇总
    total_elapsed = time.time() - start_time
    n_pass = sum(1 for r in results.values() if r["success"])
    n_fail = sum(1 for r in results.values() if not r["success"])

    print(f"\n{'=' * 60}")
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total: {len(results)}")
    print(f"  Pass:  {n_pass}")
    print(f"  Fail:  {n_fail}")
    print(f"  Time:  {total_elapsed:.1f}s")

    if n_fail > 0:
        print(f"\n  Failed experiments:")
        for exp_id, r in results.items():
            if not r["success"]:
                print(f"    [{exp_id:02d}] {EXPERIMENTS[exp_id][1]}")

    # 列出生成的结果文件
    print(f"\nGenerated files:")
    if os.path.exists(args.output):
        for f in sorted(os.listdir(args.output)):
            size_kb = os.path.getsize(os.path.join(args.output, f)) / 1024
            print(f"  {f} ({size_kb:.1f} KB)")

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
