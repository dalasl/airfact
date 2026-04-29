"""
实验 08: 自洽性采样数影响实验（论文表 5-13）

分析 MC-Dropout 推理轮次 T 对自洽性校验的影响:
  T = 1, 3, 5, 7, 9

指标: 一致性比率 ρ, 置信度 c̄, 人工审核率, 分级准确率

用法:
    python scripts/exp_08_consistency_sampling.py --output results/
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ch3_sensitive_grading.self_consistency import SelfConsistencyVerifier


# 模拟 LLM 推理
def mock_llm_inference(content, seed_offset=0):
    """模拟 LLM 推理输出（JSON 格式）"""
    rng = np.random.RandomState((hash(content) + seed_offset) % 2**31)
    content_lower = content.lower()

    if any(kw in content_lower for kw in ["绝密", "top secret", "密钥", "private key", "ssn"]):
        primary = "L4"
        probs = [0.02, 0.03, 0.10, 0.85]
    elif any(kw in content_lower for kw in ["机密", "confidential", "财务", "薪酬", "credit card"]):
        primary = "L3"
        probs = [0.03, 0.07, 0.80, 0.10]
    elif any(kw in content_lower for kw in ["内部", "internal", "会议纪要"]):
        primary = "L2"
        probs = [0.05, 0.78, 0.12, 0.05]
    else:
        primary = "L1"
        probs = [0.82, 0.10, 0.05, 0.03]

    # 按概率采样（模拟 MC-Dropout 随机性）
    levels = ["L1", "L2", "L3", "L4"]
    sampled = rng.choice(levels, p=probs)
    confidence = probs[levels.index(sampled)] + rng.uniform(-0.05, 0.05)
    confidence = max(0.3, min(0.99, confidence))

    return json.dumps({
        "level": sampled,
        "confidence": round(confidence, 3),
        "reason": f"Mock inference: {sampled}",
        "category": "test_document",
    })


TEST_CONTENTS = [
    ("公开新闻稿，媒体发布通知。", "L1"),
    ("Public product documentation.", "L1"),
    ("内部会议纪要，Q3进度。", "L2"),
    ("Internal: team schedule.", "L2"),
    ("机密财务报告：收入12M。confidential.", "L3"),
    ("Confidential client list.", "L3"),
    ("薪酬调整方案，salary details。", "L3"),
    ("绝密：密钥 private key: sk-xxx。", "L4"),
    ("Top secret: SSN 123-45-6789.", "L4"),
    ("核心机密文件，passport number。", "L4"),
]


def main():
    parser = argparse.ArgumentParser(description="实验08: 自洽性采样数影响")
    parser.add_argument("--output", default="results/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("实验 08: 自洽性采样数影响 (表 5-13)")
    print("=" * 60)

    T_values = [1, 3, 5, 7, 9]
    results = {}

    print(f"\n{'T':>4} {'Acc':>6} {'rho_avg':>7} {'c_avg':>7} {'ReviewRate':>11} {'U_avg':>7}")
    print("-" * 50)

    for T in T_values:
        verifier = SelfConsistencyVerifier(T=T, rho_min=0.6, c_min=0.85)

        y_true, y_pred = [], []
        rho_list, conf_list, review_list, uncert_list = [], [], [], []

        for content, expected in TEST_CONTENTS:
            y_true.append(expected)

            # 构造 mock LLM 函数
            def mock_llm_fn(prompt, seed=0, _content=content):
                return mock_llm_inference(_content, seed_offset=seed * 1000 + args.seed)

            decision = verifier.verify(mock_llm_fn, "grade this document")
            y_pred.append(decision.level)
            rho_list.append(decision.consistency_ratio)
            conf_list.append(decision.confidence)
            review_list.append(decision.needs_human_review)
            uncert_list.append(decision.uncertainty)

        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        acc = correct / len(y_true)
        rho_avg = float(np.mean(rho_list))
        conf_avg = float(np.mean(conf_list))
        review_rate = sum(review_list) / len(review_list)
        uncert_avg = float(np.mean(uncert_list))

        results[T] = {
            "accuracy": acc,
            "rho_avg": rho_avg,
            "confidence_avg": conf_avg,
            "review_rate": review_rate,
            "uncertainty_avg": uncert_avg,
        }
        print(f"{T:>4} {acc:>6.3f} {rho_avg:>7.3f} {conf_avg:>7.3f} {review_rate:>11.1%} {uncert_avg:>7.3f}")

    # 保存
    csv_path = os.path.join(args.output, "table_5_13_consistency_sampling.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["T", "accuracy", "rho_avg", "confidence_avg",
                                                "review_rate", "uncertainty_avg"])
        writer.writeheader()
        for T, r in results.items():
            writer.writerow({"T": T, **{k: f"{v:.4f}" for k, v in r.items()}})

    json_path = os.path.join(args.output, "exp_08_consistency_sampling.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in results.items()}, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {csv_path}")
    return results


if __name__ == "__main__":
    main()
