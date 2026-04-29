"""
RVL-CDIP 数据集子集提取与敏感等级标注

从 RVL-CDIP (400K 灰度 TIFF) 中筛选 10 类企业文档共 3000 份，
按业务语义映射四级敏感等级 (L1-L4)。

数据源: D:/datasets/rvl-cdip.tar/rvl-cdip/
输出:   data/rvlcdip/ + labels.csv
"""

import csv
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# RVL-CDIP 原始 16 类
RVLCDIP_CLASSES = {
    0: "letter", 1: "form", 2: "email", 3: "handwritten",
    4: "advertisement", 5: "scientific_report", 6: "scientific_publication",
    7: "specification", 8: "file_folder", 9: "news_article",
    10: "budget", 11: "invoice", 12: "presentation",
    13: "questionnaire", 14: "resume", 15: "memo",
}

# 筛选的 10 类及其到 L1-L4 的映射
CATEGORY_TO_SENSITIVITY: Dict[int, str] = {
    # L1 公开: 科学文献、文件夹封面、科学报告
    5: "L1",   # scientific_report
    6: "L1",   # scientific_publication
    8: "L1",   # file_folder
    # L2 内部: 信件、备忘录、邮件
    0: "L2",   # letter
    2: "L2",   # email
    15: "L2",  # memo
    # L3 机密: 发票、预算
    10: "L3",  # budget
    11: "L3",  # invoice
    # L4 绝密: 规范说明、表格
    1: "L4",   # form
    7: "L4",   # specification
}

# 各等级目标采样数 (总计 3000)
TARGET_DISTRIBUTION: Dict[str, int] = {
    "L1": 750,
    "L2": 900,
    "L3": 800,
    "L4": 550,
}

# 各等级内各类别的采样比例
_LEVEL_CATEGORY_QUOTAS: Dict[str, Dict[int, int]] = {
    "L1": {5: 250, 6: 250, 8: 250},
    "L2": {0: 300, 2: 300, 15: 300},
    "L3": {10: 400, 11: 400},
    "L4": {1: 275, 7: 275},
}


def _parse_label_file(label_path: str) -> Dict[int, List[str]]:
    """解析 RVL-CDIP 标签文件，返回 {category_id: [image_paths]}"""
    category_files: Dict[int, List[str]] = {i: [] for i in range(16)}
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(" ", 1)
            if len(parts) != 2:
                continue
            img_path, cat_str = parts
            cat_id = int(cat_str)
            category_files[cat_id].append(img_path)
    return category_files


def extract_rvlcdip_subset(
    dataset_root: str = "D:/datasets/rvl-cdip.tar/rvl-cdip",
    output_dir: str = "data/rvlcdip",
    seed: int = 42,
    copy_images: bool = True,
) -> str:
    """从 RVL-CDIP 提取 3000 份子集并标注敏感等级

    Args:
        dataset_root: RVL-CDIP 数据集根目录
        output_dir: 输出目录
        seed: 随机种子
        copy_images: 是否复制图像文件（False 仅生成标签）

    Returns:
        生成的 labels.csv 路径
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    images_dir = os.path.join(dataset_root, "images")
    labels_dir = os.path.join(dataset_root, "labels")

    # 合并 train/val/test 标签
    all_category_files: Dict[int, List[str]] = {i: [] for i in range(16)}
    for split in ["train.txt", "val.txt", "test.txt"]:
        label_path = os.path.join(labels_dir, split)
        if not os.path.exists(label_path):
            continue
        parsed = _parse_label_file(label_path)
        for cat_id, paths in parsed.items():
            all_category_files[cat_id].extend(paths)

    # 按配额采样
    selected: List[Tuple[str, int, str, str]] = []  # (src_path, cat_id, cat_name, level)

    for level, cat_quotas in _LEVEL_CATEGORY_QUOTAS.items():
        for cat_id, quota in cat_quotas.items():
            available = all_category_files[cat_id]
            if len(available) < quota:
                print(
                    f"[WARN] 类别 {RVLCDIP_CLASSES[cat_id]} 可用 {len(available)} 份，"
                    f"不足配额 {quota}，全部采用"
                )
                sampled = available
            else:
                sampled = random.sample(available, quota)

            for rel_path in sampled:
                selected.append((
                    rel_path,
                    cat_id,
                    RVLCDIP_CLASSES[cat_id],
                    level,
                ))

    random.shuffle(selected)

    # 写入 labels.csv 并复制图像
    csv_path = os.path.join(output_dir, "labels.csv")
    copied = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "doc_id", "original_path", "local_filename",
            "category_id", "category_name", "sensitivity_level",
        ])
        for idx, (rel_path, cat_id, cat_name, level) in enumerate(selected):
            doc_id = f"rvlcdip_{idx:05d}"
            # 标准化文件名
            ext = Path(rel_path).suffix or ".tif"
            local_fname = f"{doc_id}{ext}"

            if copy_images:
                src_full = os.path.join(dataset_root, rel_path)
                if not os.path.exists(src_full):
                    # 尝试在 images/ 子目录下查找
                    src_full = os.path.join(images_dir, rel_path)
                dst_full = os.path.join(output_dir, local_fname)
                if os.path.exists(src_full):
                    shutil.copy2(src_full, dst_full)
                    copied += 1
                else:
                    print(f"[WARN] 源文件不存在: {src_full}")

            writer.writerow([doc_id, rel_path, local_fname, cat_id, cat_name, level])

    # 统计
    level_counts = {}
    for _, _, _, level in selected:
        level_counts[level] = level_counts.get(level, 0) + 1

    print(f"[RVL-CDIP] 提取完成:")
    print(f"  总计: {len(selected)} 份")
    print(f"  分布: {level_counts}")
    if copy_images:
        print(f"  复制图像: {copied} 份")
    print(f"  标签文件: {csv_path}")
    return csv_path


def get_rvlcdip_stats(csv_path: str) -> Dict:
    """读取 labels.csv 并返回统计信息"""
    stats = {
        "total": 0,
        "by_level": {"L1": 0, "L2": 0, "L3": 0, "L4": 0},
        "by_category": {},
    }
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["total"] += 1
            level = row["sensitivity_level"]
            stats["by_level"][level] = stats["by_level"].get(level, 0) + 1
            cat = row["category_name"]
            stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RVL-CDIP 子集提取")
    parser.add_argument(
        "--dataset-root",
        default="D:/datasets/rvl-cdip.tar/rvl-cdip",
        help="RVL-CDIP 数据集根目录",
    )
    parser.add_argument(
        "--output-dir",
        default="data/rvlcdip",
        help="输出目录",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-copy", action="store_true",
        help="仅生成标签，不复制图像",
    )
    args = parser.parse_args()

    csv_path = extract_rvlcdip_subset(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        seed=args.seed,
        copy_images=not args.no_copy,
    )
    stats = get_rvlcdip_stats(csv_path)
    print(f"\n统计: {stats}")
