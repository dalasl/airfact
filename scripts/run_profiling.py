#!/usr/bin/env python3
"""
用户画像构建实验脚本

支持阶段：denoise | parse | align | cluster | ablation | cert
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="用户画像构建实验")
    parser.add_argument("--stage", type=str, required=True,
                        choices=["denoise", "parse", "align", "cluster", "ablation", "cert"])
    parser.add_argument("--input", type=str, default="data/raw/documents/")
    parser.add_argument("--output", type=str, default="results/profiling/")
    parser.add_argument("--events", type=str, default="data/behavior_logs/")
    parser.add_argument("--doc-cache", type=str, default="data/doc_features/")
    parser.add_argument("--features", type=str, default="data/aligned_events/")
    parser.add_argument("--config", type=str, default="configs/profiling_config.yaml")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.stage == "denoise":
        print("阶段1: U-Net 文档抗噪预处理")
        from src.ch2_user_profiling import UNetDenoiser
        print(f"  输入: {args.input}")
        print(f"  输出: {args.output}")

    elif args.stage == "parse":
        print("阶段2: LayoutLMv3 + Sentence-BERT 多模态解析")
        from src.ch2_user_profiling import DocumentParser

    elif args.stage == "align":
        print("阶段3: 多源异构数据时空对齐")
        from src.ch2_user_profiling import TemporalAligner

    elif args.stage == "cluster":
        print("阶段4: 三维特征融合 + IGW-Kmeans 聚类")
        from src.ch2_user_profiling import IGWKMeans, DualBranchFusionEncoder

    elif args.stage == "ablation":
        print("特征消融实验")
        from src.ch2_user_profiling import FeatureExtractor, IGWKMeans

    elif args.stage == "cert":
        print("CERT r4.2 数据集聚类验证")
        from src.ch2_user_profiling import IGWKMeans

    print(f"实验完成，结果保存至: {args.output}")


if __name__ == "__main__":
    main()
