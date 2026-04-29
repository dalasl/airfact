"""
实验 11: 端到端泄露检测对比实验（论文表 5-4, 5-5）

表 5-4: 端到端检测多基线对比
  1. Static-Rules (传统 DLP)
  2. DeBERTa-ft + Static-Rules
  3. Presidio-NER + Static-Rules
  4. Proposed (动态画像驱动)

表 5-5: 分通道检测结果
  - Email, HTTP/Cloud, USB/Device, IM

用法:
    python scripts/exp_11_e2e_detection.py --output results/
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines.static_rules import StaticRuleGrader
from src.baselines.deberta_classifier import DeBERTaClassifier
from src.baselines.presidio_grading import PresidioGrader
from src.ch4_rule_generation.adaptive_decision import (
    AdaptiveDecisionEngine,
    OperationEvent,
    ResponseAction,
)


def generate_detection_scenarios(n=200, seed=42):
    """生成端到端检测场景

    每个场景包含: 用户操作 + 文档内容 + 真实标签(leak/normal)
    """
    rng = np.random.RandomState(seed)
    scenarios = []

    channels = ["email", "http", "usb", "im"]
    contents_normal = [
        ("公开新闻稿，媒体发布通知。", "/public/news.docx"),
        ("Internal meeting notes, project update.", "/internal/notes.docx"),
        ("Public product manual v2.0.", "/public/manual.pdf"),
        ("内部培训材料，新员工入职指南。", "/internal/training.pdf"),
    ]
    contents_leak = [
        ("机密财务报告：收入12M，净利润3.5M。confidential.", "/finance/report.xlsx"),
        ("Top secret: API key sk-prod-xxx. SSN: 123-45-6789.", "/secret/creds.txt"),
        ("薪酬调整方案：加薪5%-15%，salary details.", "/hr/salary.xlsx"),
        ("客户名单含联系方式，credit card 4111-1111-1111-1111.", "/crm/clients.xlsx"),
        ("绝密：数据库密钥 private key。护照号 E12345678。", "/secret/keys.env"),
        ("合同金额230万，机密文件。confidential contract.", "/legal/contract.pdf"),
    ]

    for i in range(n):
        channel = rng.choice(channels)
        is_leak = rng.random() < 0.4  # 40% 泄露场景

        if is_leak:
            content, path = contents_leak[rng.randint(0, len(contents_leak))]
            action = rng.choice(["copy", "send", "upload"])
            is_work = rng.random() < 0.3
            net_env = rng.choice(["internal", "external"])
        else:
            content, path = contents_normal[rng.randint(0, len(contents_normal))]
            action = rng.choice(["read", "write"])
            is_work = rng.random() < 0.85
            net_env = "internal"

        device = "usb" if channel == "usb" else "local"
        event = OperationEvent(
            user_id=f"user_{i % 30}", action=action, file_path=path,
            process_name="app.exe", network_env=net_env, device_type=device,
            is_work_hours=is_work,
        )

        scenarios.append({
            "event": event,
            "content": content,
            "channel": channel,
            "is_leak": is_leak,
        })

    return scenarios


def detect_static_rules(scenario, grader):
    result = grader.grade(content=scenario["content"],
                          file_path=scenario["event"].file_path,
                          channel=scenario["channel"])
    return result.level in ("L3", "L4")


def detect_deberta_rules(scenario, deberta, grader):
    dr = deberta._mock_grade(scenario["content"])
    sr = grader.grade(content=scenario["content"],
                      file_path=scenario["event"].file_path,
                      channel=scenario["channel"])
    combined_level = max(dr.level, sr.level)
    return combined_level in ("L3", "L4")


def detect_presidio_rules(scenario, presidio, grader):
    pr = presidio.grade(scenario["content"])
    sr = grader.grade(content=scenario["content"],
                      file_path=scenario["event"].file_path,
                      channel=scenario["channel"])
    combined_level = max(pr.level, sr.level)
    return combined_level in ("L3", "L4")


def detect_proposed(scenario, engine, grader):
    sr = grader.grade(content=scenario["content"],
                      file_path=scenario["event"].file_path,
                      channel=scenario["channel"])
    rng = np.random.RandomState(hash(scenario["content"]) % 2**31)
    operator_profile = rng.randn(896) * 0.5
    owner_profile = rng.randn(896) * 0.5
    behavior_feature = rng.randn(896) * 0.3
    decision = engine.decide(scenario["event"], operator_profile, owner_profile, behavior_feature)
    # 融合: 分级结果 + 行为检测
    is_detected = (sr.level in ("L3", "L4") and
                   decision.action != ResponseAction.SILENT_MONITOR)
    return is_detected


def compute_detection_metrics(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)

    recall = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    fpr = fp / (fp + tn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"recall": recall, "precision": precision, "fpr": fpr, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main():
    parser = argparse.ArgumentParser(description="实验11: 端到端检测")
    parser.add_argument("--output", default="results/")
    parser.add_argument("--n_scenarios", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("实验 11: 端到端泄露检测对比 (表 5-4, 5-5)")
    print("=" * 60)

    scenarios = generate_detection_scenarios(n=args.n_scenarios, seed=args.seed)
    print(f"场景数: {len(scenarios)}")
    print(f"泄露比例: {sum(1 for s in scenarios if s['is_leak']) / len(scenarios):.1%}")

    grader = StaticRuleGrader()
    deberta = DeBERTaClassifier()
    presidio = PresidioGrader()
    engine = AdaptiveDecisionEngine()

    y_true = [s["is_leak"] for s in scenarios]

    detectors = {
        "Static-Rules": lambda s: detect_static_rules(s, grader),
        "DeBERTa+Rules": lambda s: detect_deberta_rules(s, deberta, grader),
        "Presidio+Rules": lambda s: detect_presidio_rules(s, presidio, grader),
        "Proposed": lambda s: detect_proposed(s, engine, grader),
    }

    # --- 表 5-4: 总体对比 ---
    print(f"\n[表 5-4] 端到端检测对比:")
    print(f"{'Method':<16} {'Recall':>8} {'Prec':>8} {'FPR':>8} {'F1':>8}")
    print("-" * 52)

    overall_results = {}
    for method_name, detect_fn in detectors.items():
        y_pred = [detect_fn(s) for s in scenarios]
        metrics = compute_detection_metrics(y_true, y_pred)
        overall_results[method_name] = metrics
        print(f"{method_name:<16} {metrics['recall']:>8.3f} {metrics['precision']:>8.3f} "
              f"{metrics['fpr']:>8.3f} {metrics['f1']:>8.3f}")

    # --- 表 5-5: 分通道检测 ---
    print(f"\n[表 5-5] 分通道检测结果 (Proposed):")
    channels = ["email", "http", "usb", "im"]
    channel_results = {}

    print(f"{'Channel':<10} {'Recall':>8} {'Prec':>8} {'FPR':>8} {'F1':>8} {'N':>5}")
    print("-" * 45)

    for ch in channels:
        ch_scenarios = [s for s in scenarios if s["channel"] == ch]
        if not ch_scenarios:
            continue
        ch_true = [s["is_leak"] for s in ch_scenarios]
        ch_pred = [detect_proposed(s, engine, grader) for s in ch_scenarios]
        metrics = compute_detection_metrics(ch_true, ch_pred)
        metrics["n"] = len(ch_scenarios)
        channel_results[ch] = metrics
        print(f"{ch:<10} {metrics['recall']:>8.3f} {metrics['precision']:>8.3f} "
              f"{metrics['fpr']:>8.3f} {metrics['f1']:>8.3f} {metrics['n']:>5}")

    # 保存
    csv_path = os.path.join(args.output, "table_5_4_e2e_detection.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "recall", "precision", "fpr", "f1"])
        writer.writeheader()
        for method, m in overall_results.items():
            writer.writerow({"method": method, **{k: f"{v:.4f}" for k, v in m.items()
                                                   if k in ("recall", "precision", "fpr", "f1")}})

    csv_path2 = os.path.join(args.output, "table_5_5_channel_detection.csv")
    with open(csv_path2, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["channel", "recall", "precision", "fpr", "f1", "n"])
        writer.writeheader()
        for ch, m in channel_results.items():
            writer.writerow({"channel": ch, **{k: f"{v:.4f}" if isinstance(v, float) else v
                                                for k, v in m.items()
                                                if k in ("recall", "precision", "fpr", "f1", "n")}})

    json_path = os.path.join(args.output, "exp_11_e2e_detection.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"overall": overall_results, "channel": channel_results}, f, indent=2)

    print(f"\n结果已保存: {csv_path}, {csv_path2}")
    return {"overall": overall_results, "channel": channel_results}


if __name__ == "__main__":
    main()
