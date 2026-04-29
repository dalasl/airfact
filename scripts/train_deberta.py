"""
DeBERTa-v3-base 基线训练脚本

论文 5.1.2 实验配置：
    模型: microsoft/deberta-v3-base
    任务: L1-L4 四分类
    超参数: lr=2e-5, batch=16, epoch=5, AdamW, 10% linear warmup
    数据: data/ 下的标注文档

用法:
    python scripts/train_deberta.py --data_dir data --output_dir checkpoints/deberta
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_documents(data_dir: str):
    """加载标注文档

    从 labels.csv 和对应目录加载文档文本及标签。
    """
    import csv

    texts, labels = [], []

    for subdir in ["rvlcdip", "enron_pdf", "custom"]:
        label_file = os.path.join(data_dir, subdir, "labels.csv")
        if not os.path.exists(label_file):
            continue

        with open(label_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 尝试读取文本
                text_path = os.path.join(data_dir, subdir, row.get("text_file", ""))
                if os.path.exists(text_path):
                    with open(text_path, "r", encoding="utf-8", errors="ignore") as tf:
                        text = tf.read()[:2048]
                else:
                    text = row.get("content", row.get("doc_summary", ""))

                if text and row.get("level"):
                    texts.append(text)
                    labels.append(row["level"])

    return texts, labels


def main():
    parser = argparse.ArgumentParser(description="DeBERTa 基线训练")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="checkpoints/deberta")
    parser.add_argument("--model_name", default="microsoft/deberta-v3-base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("DeBERTa-v3-base Baseline Training")
    print("=" * 60)

    # 加载数据
    texts, labels = load_documents(args.data_dir)
    if not texts:
        print(f"No documents found in {args.data_dir}")
        print("Expected structure: {data_dir}/{{rvlcdip,enron_pdf,custom}}/labels.csv")
        print("Generating synthetic training data for demo...")

        texts = [
            "This is a public press release about our new product launch.",
            "Internal memo: Q3 team meeting schedule and action items.",
            "Confidential financial report: Q1 revenue 12M, expenses 8.5M.",
            "Top secret: API keys and database credentials for production.",
        ] * 25
        labels = ["L1", "L2", "L3", "L4"] * 25

    print(f"Total documents: {len(texts)}")
    from collections import Counter
    print(f"Label distribution: {dict(Counter(labels))}")

    # 划分训练/验证集
    import random
    random.seed(args.seed)
    indices = list(range(len(texts)))
    random.shuffle(indices)
    val_size = int(len(indices) * args.val_ratio)

    val_indices = set(indices[:val_size])
    train_texts = [texts[i] for i in indices if i not in val_indices]
    train_labels = [labels[i] for i in indices if i not in val_indices]
    val_texts = [texts[i] for i in indices if i in val_indices]
    val_labels = [labels[i] for i in indices if i in val_indices]

    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}")

    # 训练
    from src.baselines.deberta_classifier import DeBERTaClassifier

    classifier = DeBERTaClassifier(
        model_name=args.model_name,
        max_length=args.max_length,
    )

    history = classifier.train(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
    )

    # 保存训练历史
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Checkpoints saved to {args.output_dir}")
    for epoch, (tl, vl, va) in enumerate(zip(
        history["train_loss"],
        history.get("val_loss", []),
        history.get("val_accuracy", []),
    )):
        print(f"  Epoch {epoch+1}: train_loss={tl:.4f}, val_loss={vl:.4f}, val_acc={va:.4f}")


if __name__ == "__main__":
    main()
