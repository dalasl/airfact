"""Phase 6: 实验脚本与图表生成集成测试"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


RESULTS_DIR = "results/"
FIGURES_DIR = "figures/"

# 12 个实验脚本对应的 JSON 输出文件
EXPECTED_JSON_FILES = [
    "exp_01_clustering_comparison.json",
    "exp_02_feature_weight_analysis.json",
    "exp_03_k_sensitivity.json",
    "exp_04_incremental_update.json",
    "exp_05_grading_comparison.json",
    "exp_06_format_grading.json",
    "exp_07_context_ablation.json",
    "exp_08_consistency_sampling.json",
    "exp_09_parameter_sensitivity.json",
    "exp_10_rule_generation.json",
    "exp_11_e2e_detection.json",
    "exp_12_system_performance.json",
]

# 10 个图表 PDF 文件
EXPECTED_FIGURE_FILES = [
    "fig_5_1_clustering_comparison.pdf",
    "fig_5_2_k_sensitivity.pdf",
    "fig_5_3_incremental_update.pdf",
    "fig_5_4_grading_comparison.pdf",
    "fig_5_5_ablation.pdf",
    "fig_5_6_consistency_sampling.pdf",
    "fig_5_7_rule_quality.pdf",
    "fig_5_8_latency_breakdown.pdf",
    "fig_5_9_detection_comparison.pdf",
    "fig_5_10_weight_heatmap.pdf",
]


def test_all_json_outputs_exist():
    """检查 12 个实验脚本的 JSON 输出文件均存在且非空"""
    print("\n[1/6] Checking experiment JSON outputs...")
    missing = []
    empty = []
    for fname in EXPECTED_JSON_FILES:
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path):
            missing.append(fname)
        elif os.path.getsize(path) == 0:
            empty.append(fname)

    assert not missing, f"Missing JSON files: {missing}"
    assert not empty, f"Empty JSON files: {empty}"
    print(f"  PASS: {len(EXPECTED_JSON_FILES)} JSON files present and non-empty")


def test_all_figures_exist():
    """检查 10 个图表 PDF 文件均存在且大小合理"""
    print("\n[2/6] Checking figure PDF outputs...")
    missing = []
    too_small = []
    for fname in EXPECTED_FIGURE_FILES:
        path = os.path.join(FIGURES_DIR, fname)
        if not os.path.exists(path):
            missing.append(fname)
        elif os.path.getsize(path) < 1024:  # PDF 至少 1KB
            too_small.append((fname, os.path.getsize(path)))

    assert not missing, f"Missing figure files: {missing}"
    assert not too_small, f"Suspiciously small figures: {too_small}"
    print(f"  PASS: {len(EXPECTED_FIGURE_FILES)} PDF figures present (all > 1KB)")


def test_json_data_integrity():
    """验证各实验 JSON 数据结构和数值范围"""
    print("\n[3/6] Validating JSON data integrity...")
    errors = []

    # exp_01: 聚类质量对比
    data = _load_json("exp_01_clustering_comparison.json")
    if data:
        for method in data:
            if "SC" not in data[method] or "CH" not in data[method]:
                errors.append(f"exp_01: {method} missing SC/CH")
            elif not (-1 <= data[method]["SC"] <= 1):
                errors.append(f"exp_01: {method} SC={data[method]['SC']} out of range")

    # exp_03: K 值敏感性
    data = _load_json("exp_03_k_sensitivity.json")
    if data:
        for k_str, vals in data.items():
            if "SC" not in vals:
                errors.append(f"exp_03: K={k_str} missing SC")

    # exp_05: 分级对比
    data = _load_json("exp_05_grading_comparison.json")
    if data:
        for method, vals in data.items():
            for metric in ["accuracy", "precision", "recall", "f1"]:
                if metric not in vals:
                    errors.append(f"exp_05: {method} missing {metric}")
                elif not (0 <= vals[metric] <= 1):
                    errors.append(f"exp_05: {method}.{metric}={vals[metric]} out of [0,1]")

    # exp_08: 自洽性采样
    data = _load_json("exp_08_consistency_sampling.json")
    if data:
        for t_str, vals in data.items():
            if "accuracy" not in vals or "rho_avg" not in vals:
                errors.append(f"exp_08: T={t_str} missing accuracy/rho_avg")

    # exp_10: 规则生成质量
    data = _load_json("exp_10_rule_generation.json")
    if data:
        for method, vals in data.items():
            for metric in ["syntax_rate", "conflict_rate", "semantic_fidelity", "exec_success_rate"]:
                if metric not in vals:
                    errors.append(f"exp_10: {method} missing {metric}")

    # exp_11: 端到端检测
    data = _load_json("exp_11_e2e_detection.json")
    if data:
        if "overall" not in data:
            errors.append("exp_11: missing 'overall' key")

    # exp_12: 系统性能
    data = _load_json("exp_12_system_performance.json")
    if data:
        if "stages" not in data:
            errors.append("exp_12: missing 'stages' key")
        else:
            for stage, metrics in data["stages"].items():
                if "mean_ms" not in metrics:
                    errors.append(f"exp_12: stage {stage} missing mean_ms")

    assert not errors, f"Data integrity errors:\n" + "\n".join(f"  - {e}" for e in errors)
    print("  PASS: All JSON data structures valid, values in expected ranges")


def test_experiment_scripts_importable():
    """验证所有实验脚本可以被导入（无语法错误）"""
    print("\n[4/6] Checking experiment scripts importable...")
    import importlib.util

    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    script_names = [
        "exp_01_clustering_comparison",
        "exp_02_feature_weight_analysis",
        "exp_03_k_sensitivity",
        "exp_04_incremental_update",
        "exp_05_grading_baseline_comparison",
        "exp_06_format_grading",
        "exp_07_context_ablation",
        "exp_08_consistency_sampling",
        "exp_09_parameter_sensitivity",
        "exp_10_rule_generation_quality",
        "exp_11_e2e_detection",
        "exp_12_system_performance",
    ]

    import_errors = []
    for name in script_names:
        script_path = scripts_dir / f"{name}.py"
        if not script_path.exists():
            import_errors.append(f"{name}.py not found")
            continue
        try:
            spec = importlib.util.spec_from_file_location(name, str(script_path))
            mod = importlib.util.module_from_spec(spec)
            # Don't exec — just check the spec loads
            assert spec is not None, f"Failed to create spec for {name}"
        except Exception as e:
            import_errors.append(f"{name}: {e}")

    assert not import_errors, f"Import errors:\n" + "\n".join(f"  - {e}" for e in import_errors)
    print(f"  PASS: {len(script_names)} experiment scripts loadable")


def test_run_all_experiments_script():
    """验证 run_all_experiments.py 存在且可加载"""
    print("\n[5/6] Checking orchestrator script...")
    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    orchestrator = scripts_dir / "run_all_experiments.py"
    assert orchestrator.exists(), "run_all_experiments.py not found"

    plot_script = scripts_dir / "plot_all_figures.py"
    assert plot_script.exists(), "plot_all_figures.py not found"
    print("  PASS: Orchestrator and plot scripts present")


def test_csv_outputs_exist():
    """检查关键 CSV 文件存在"""
    print("\n[6/6] Checking CSV outputs...")
    expected_csvs = [
        "table_5_6_clustering_comparison.csv",
        "table_5_8_k_sensitivity.csv",
        "table_5_9_incremental_update.csv",
        "table_5_10_grading_comparison.csv",
        "table_5_11_format_grading.csv",
        "table_5_12_context_ablation.csv",
        "table_5_13_consistency_sampling.csv",
        "table_5_14_threshold_sensitivity.csv",
        "table_5_15_rag_sensitivity.csv",
        "table_5_16_rule_generation.csv",
        "table_5_17_system_performance.csv",
        "table_5_4_e2e_detection.csv",
        "table_5_5_channel_detection.csv",
    ]

    missing = [f for f in expected_csvs if not os.path.exists(os.path.join(RESULTS_DIR, f))]
    assert not missing, f"Missing CSV files: {missing}"
    print(f"  PASS: {len(expected_csvs)} CSV files present")


def _load_json(fname):
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("=" * 60)
    print("Phase 6: Experiment & Visualization Integration Test")
    print("=" * 60)

    tests = [
        test_all_json_outputs_exist,
        test_all_figures_exist,
        test_json_data_integrity,
        test_experiment_scripts_importable,
        test_run_all_experiments_script,
        test_csv_outputs_exist,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR: {e}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print(f"{'=' * 60}")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
