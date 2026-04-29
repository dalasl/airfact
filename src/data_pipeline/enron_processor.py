"""
Enron 邮件数据集处理器

从 Pile-Enron_Emails JSONL 中解析邮件文本，
按 privilege/confidential 等敏感关键词筛选 700 封，
使用 reportlab 渲染为 PDF 图像，生成标注 CSV。

数据源: D:/datasets/Pile-Enron_Emails/raw/{train,test,val}/*.jsonl
输出:   data/enron_pdf/ + labels.csv
"""

import csv
import json
import os
import random
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 敏感关键词分级规则
# 按关键词组合判定敏感等级，同一封邮件命中最高级别
SENSITIVITY_RULES: List[Tuple[str, List[str], float]] = [
    # (级别, 关键词列表, 命中权重阈值)
    # L4: 法律特权 / 高度机密
    ("L4", [
        "attorney-client privilege", "attorney client privilege",
        "work product", "legally privileged",
        "trade secret", "strictly confidential",
        "do not distribute", "do not forward",
        "top secret", "classified information",
    ], 1),
    # L3: 商业机密 / 财务
    ("L3", [
        "confidential", "proprietary",
        "non-disclosure", "nda",
        "financial statement", "earnings",
        "merger", "acquisition",
        "insider", "material non-public",
        "restricted", "internal only",
    ], 1),
    # L2: 内部沟通
    ("L2", [
        "internal", "private",
        "draft", "preliminary",
        "not for external", "for internal use",
        "meeting minutes", "action items",
    ], 1),
]

# 各级别目标采样数 (总计 700)
TARGET_DISTRIBUTION: Dict[str, int] = {
    "L1": 180,
    "L2": 220,
    "L3": 180,
    "L4": 120,
}


def _classify_email(content: str) -> str:
    """根据内容关键词判定邮件敏感等级"""
    lower = content.lower()
    for level, keywords, _ in SENSITIVITY_RULES:
        if any(kw in lower for kw in keywords):
            return level
    return "L1"


def _scan_jsonl_files(data_root: str) -> List[Dict]:
    """扫描所有 JSONL 文件，返回邮件记录列表"""
    records = []
    for split in ["train", "test", "val"]:
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            continue
        for fname in sorted(os.listdir(split_dir)):
            if not fname.endswith(".jsonl"):
                continue
            fpath = os.path.join(split_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        content = rec.get("content", "")
                        # 过滤过短的邮件（< 50 字符无法有效分级）
                        if len(content) < 50:
                            continue
                        records.append(rec)
                    except json.JSONDecodeError:
                        continue
    return records


def _render_email_to_pdf(
    content: str,
    output_path: str,
    doc_id: str,
    page_width: float = 595.27,  # A4
    page_height: float = 841.89,
) -> bool:
    """将邮件文本渲染为 PDF 文件"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
            leftMargin=2.5 * cm,
            rightMargin=2.5 * cm,
        )

        styles = getSampleStyleSheet()
        # 邮件标题样式
        header_style = ParagraphStyle(
            "EmailHeader",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            textColor="grey",
        )
        # 正文样式
        body_style = ParagraphStyle(
            "EmailBody",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
        )

        story = []

        # 添加文档 ID 水印（模拟企业文档管理系统标识）
        story.append(Paragraph(f"Document ID: {doc_id}", header_style))
        story.append(Spacer(1, 0.5 * cm))

        # 处理邮件内容：按段落分割
        paragraphs = content.split("\n\n")
        for para_text in paragraphs:
            para_text = para_text.strip()
            if not para_text:
                continue
            # 转义 XML 特殊字符
            safe_text = (
                para_text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br/>")
            )
            story.append(Paragraph(safe_text, body_style))
            story.append(Spacer(1, 0.3 * cm))

        if not story:
            return False

        doc.build(story)
        return True
    except Exception as e:
        print(f"[WARN] PDF 渲染失败 ({doc_id}): {e}")
        return False


def extract_enron_subset(
    data_root: str = "D:/datasets/Pile-Enron_Emails/raw",
    output_dir: str = "data/enron_pdf",
    seed: int = 42,
    render_pdf: bool = True,
    max_scan: int = 50000,
) -> str:
    """从 Enron 邮件数据集提取 700 封并渲染为 PDF

    Args:
        data_root: Enron 数据集根目录
        output_dir: 输出目录
        seed: 随机种子
        render_pdf: 是否渲染 PDF（False 仅生成标签）
        max_scan: 最大扫描邮件数

    Returns:
        生成的 labels.csv 路径
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print("[Enron] 扫描 JSONL 文件...")
    all_records = _scan_jsonl_files(data_root)
    print(f"[Enron] 扫描到 {len(all_records)} 条有效邮件")

    # 截断避免处理过多
    if len(all_records) > max_scan:
        random.shuffle(all_records)
        all_records = all_records[:max_scan]

    # 按敏感等级分桶
    buckets: Dict[str, List[Dict]] = {"L1": [], "L2": [], "L3": [], "L4": []}
    for rec in all_records:
        level = _classify_email(rec.get("content", ""))
        buckets[level].append(rec)

    print("[Enron] 分级统计:")
    for level in ["L1", "L2", "L3", "L4"]:
        print(f"  {level}: {len(buckets[level])} 封")

    # 按配额采样
    selected: List[Tuple[Dict, str]] = []
    for level, target in TARGET_DISTRIBUTION.items():
        available = buckets[level]
        if len(available) < target:
            print(
                f"[WARN] {level} 可用 {len(available)} 封，"
                f"不足配额 {target}，全部采用"
            )
            sampled = available
        else:
            sampled = random.sample(available, target)
        for rec in sampled:
            selected.append((rec, level))

    random.shuffle(selected)

    # 写入 labels.csv 并渲染 PDF
    csv_path = os.path.join(output_dir, "labels.csv")
    rendered = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "doc_id", "source_id", "pdf_filename",
            "sensitivity_level", "content_length", "has_attachment_ref",
        ])
        for idx, (rec, level) in enumerate(selected):
            doc_id = f"enron_{idx:05d}"
            content = rec.get("content", "")
            pdf_fname = f"{doc_id}.pdf"

            # 检测是否提及附件
            has_attach = bool(re.search(
                r"attach|enclos|see (the )?file",
                content.lower(),
            ))

            if render_pdf:
                pdf_path = os.path.join(output_dir, pdf_fname)
                if _render_email_to_pdf(content, pdf_path, doc_id):
                    rendered += 1

            writer.writerow([
                doc_id,
                rec.get("source_id", ""),
                pdf_fname,
                level,
                len(content),
                int(has_attach),
            ])

    print(f"\n[Enron] 提取完成:")
    print(f"  总计: {len(selected)} 封")
    level_counts = {}
    for _, level in selected:
        level_counts[level] = level_counts.get(level, 0) + 1
    print(f"  分布: {level_counts}")
    if render_pdf:
        print(f"  渲染 PDF: {rendered} 份")
    print(f"  标签文件: {csv_path}")
    return csv_path


def get_enron_stats(csv_path: str) -> Dict:
    """读取 labels.csv 并返回统计信息"""
    stats = {
        "total": 0,
        "by_level": {"L1": 0, "L2": 0, "L3": 0, "L4": 0},
        "avg_content_length": 0,
        "with_attachment_ref": 0,
    }
    total_len = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["total"] += 1
            stats["by_level"][row["sensitivity_level"]] += 1
            total_len += int(row["content_length"])
            if row["has_attachment_ref"] == "1":
                stats["with_attachment_ref"] += 1
    if stats["total"] > 0:
        stats["avg_content_length"] = total_len // stats["total"]
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enron 邮件子集提取与 PDF 渲染")
    parser.add_argument(
        "--data-root",
        default="D:/datasets/Pile-Enron_Emails/raw",
        help="Enron 数据集根目录",
    )
    parser.add_argument(
        "--output-dir",
        default="data/enron_pdf",
        help="输出目录",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-render", action="store_true",
        help="仅生成标签，不渲染 PDF",
    )
    parser.add_argument(
        "--max-scan", type=int, default=50000,
        help="最大扫描邮件数",
    )
    args = parser.parse_args()

    csv_path = extract_enron_subset(
        data_root=args.data_root,
        output_dir=args.output_dir,
        seed=args.seed,
        render_pdf=not args.no_render,
        max_scan=args.max_scan,
    )
    stats = get_enron_stats(csv_path)
    print(f"\n统计: {stats}")
