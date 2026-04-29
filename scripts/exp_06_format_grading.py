"""
实验 06: 分格式分级性能实验（论文表 5-11）

分析不同文件格式对分级精度的影响:
  - docx (Word文档)
  - pdf (PDF报告)
  - xlsx (Excel表格)
  - txt/log (纯文本)
  - eml (邮件)

指标: Accuracy, F1 (per format)

用法:
    python scripts/exp_06_format_grading.py --output results/
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

FORMAT_TEST_DOCS = {
    "docx": [
        {"content": "公开发布：产品手册更新说明。", "path": "/pub/manual.docx", "expected": "L1"},
        {"content": "内部会议纪要：Q3进度汇报，项目进度80%。通讯录附后。",
         "path": "/int/meeting.docx", "expected": "L2"},
        {"content": "机密文件：年度财务报表，收入数据。confidential.",
         "path": "/fin/report.docx", "expected": "L3"},
        {"content": "绝密：核心机密技术方案，含密钥 private key: sk-xxx。",
         "path": "/sec/plan.docx", "expected": "L4"},
    ],
    "pdf": [
        {"content": "公开产品白皮书，新闻稿配套材料。", "path": "/pub/wp.pdf", "expected": "L1"},
        {"content": "仅限内部：项目进度报告。internal only.", "path": "/int/progress.pdf", "expected": "L2"},
        {"content": "Confidential: financial audit report Q2.", "path": "/fin/audit.pdf", "expected": "L3"},
        {"content": "Top secret: production credentials and SSN: 123-45-6789.",
         "path": "/sec/creds.pdf", "expected": "L4"},
    ],
    "xlsx": [
        {"content": "公开数据：产品价目表。public pricing.", "path": "/pub/pricing.xlsx", "expected": "L1"},
        {"content": "内部数据：员工通讯录，部门联系方式。", "path": "/int/contacts.xlsx", "expected": "L2"},
        {"content": "机密：薪酬明细表，salary details per employee。",
         "path": "/hr/salary.xlsx", "expected": "L3"},
        {"content": "绝密：信用卡号 4111-1111-1111-1111, SSN 999-88-7777。",
         "path": "/sec/pii.xlsx", "expected": "L4"},
    ],
    "txt": [
        {"content": "README: This project is open source under MIT license.",
         "path": "/pub/readme.txt", "expected": "L1"},
        {"content": "Internal notes: deployment schedule for staging env.",
         "path": "/int/notes.txt", "expected": "L2"},
        {"content": "Confidential: source code review notes, architecture details.",
         "path": "/src/review.txt", "expected": "L3"},
        {"content": "Secret key: sk-prod-abc123. Passport: E12345678. 绝密。",
         "path": "/sec/keys.txt", "expected": "L4"},
    ],
    "eml": [
        {"content": "公开通知：公司活动邀请，欢迎参加。public event.",
         "path": "/mail/invite.eml", "expected": "L1"},
        {"content": "Internal: team meeting agenda, please do not forward.",
         "path": "/mail/agenda.eml", "expected": "L2"},
        {"content": "Confidential: client contract review, amount $500K.",
         "path": "/mail/contract.eml", "expected": "L3"},
        {"content": "Credit card: 4111-1111-1111-1111. SSN: 123-45-6789. Top secret.",
         "path": "/mail/pii.eml", "expected": "L4"},
    ],
}


def main():
    parser = argparse.ArgumentParser(description="实验06: 分格式分级性能")
    parser.add_argument("--output", default="results/")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("实验 06: 分格式分级性能 (表 5-11)")
    print("=" * 60)

    static_grader = StaticRuleGrader()
    deberta = DeBERTaClassifier()
    presidio = PresidioGrader()

    methods = {
        "Static-Rules": lambda c, p: static_grader.grade(content=c, file_path=p).level,
        "DeBERTa-ft": lambda c, p: deberta._mock_grade(c).level,
        "Presidio-NER": lambda c, p: presidio.grade(c).level,
    }

    results = {}

    print(f"\n{'Format':<8}", end="")
    for method in methods:
        print(f" {method:>13}", end="")
    print()
    print("-" * 50)

    for fmt, docs in FORMAT_TEST_DOCS.items():
        y_true = [d["expected"] for d in docs]
        format_results = {}

        for method_name, grade_fn in methods.items():
            y_pred = [grade_fn(d["content"], d["path"]) for d in docs]
            correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
            acc = correct / len(y_true)
            format_results[method_name] = {"accuracy": acc, "predictions": y_pred}

        results[fmt] = format_results

        print(f"{fmt:<8}", end="")
        for method_name in methods:
            acc = format_results[method_name]["accuracy"]
            print(f" {acc:>13.1%}", end="")
        print()

    # 汇总各方法的总体表现
    print(f"\n{'Overall':<8}", end="")
    for method_name in methods:
        total_correct = 0
        total_docs = 0
        for fmt in FORMAT_TEST_DOCS:
            y_true = [d["expected"] for d in FORMAT_TEST_DOCS[fmt]]
            y_pred = results[fmt][method_name]["predictions"]
            total_correct += sum(1 for t, p in zip(y_true, y_pred) if t == p)
            total_docs += len(y_true)
        overall_acc = total_correct / total_docs
        print(f" {overall_acc:>13.1%}", end="")
    print()

    # 保存 CSV
    csv_path = os.path.join(args.output, "table_5_11_format_grading.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fields = ["format"] + [f"{m}_accuracy" for m in methods]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for fmt, fmt_results in results.items():
            row = {"format": fmt}
            for method_name in methods:
                row[f"{method_name}_accuracy"] = f"{fmt_results[method_name]['accuracy']:.4f}"
            writer.writerow(row)

    json_path = os.path.join(args.output, "exp_06_format_grading.json")
    serializable = {}
    for fmt, fmt_results in results.items():
        serializable[fmt] = {m: {"accuracy": r["accuracy"]} for m, r in fmt_results.items()}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {csv_path}")
    return results


if __name__ == "__main__":
    main()
