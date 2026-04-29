#!/usr/bin/env python3
"""数据集准备脚本"""

import os

DATASETS = {
    "RVL-CDIP": {
        "url": "https://huggingface.co/datasets/rvl_cdip",
        "description": "16类企业文档扫描件，筛选10类，采样3000份",
        "license": "研究用途",
    },
    "Enron": {
        "url": "https://www.cs.cmu.edu/~enron/ (EDRM版本)",
        "description": "企业邮件语料，采样700份渲染为PDF",
        "license": "公开数据集",
    },
    "CERT r4.2": {
        "url": "https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247",
        "description": "CMU SEI 内部威胁模拟数据集",
        "license": "研究用途，需申请",
    },
}


def prepare_all():
    print("=" * 50)
    print("数据集准备")
    print("=" * 50)

    for name, info in DATASETS.items():
        print(f"\n--- {name} ---")
        print(f"  URL: {info['url']}")
        print(f"  描述: {info['description']}")
        print(f"  许可: {info['license']}")

    print("\n请按照 data/README.md 中的说明手动下载并放置数据集。")
    print("自建数据集通过 scripts/generate_selfbuilt.py 生成。")


if __name__ == "__main__":
    prepare_all()
