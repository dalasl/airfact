#!/usr/bin/env python3
"""
敏感数据分级实验脚本

支持阶段：build-index | grade | ablation
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="敏感数据分级实验")
    parser.add_argument("--stage", type=str, required=True,
                        choices=["build-index", "grade", "ablation"])
    parser.add_argument("--documents", type=str, default="data/raw/documents/")
    parser.add_argument("--profile", type=str, default="results/profiling/")
    parser.add_argument("--faiss-index", type=str, default="models/faiss_index/")
    parser.add_argument("--seed-cases", type=str, default="data/raw/seed_cases/")
    parser.add_argument("--config", type=str, default="configs/grading_config.yaml")
    parser.add_argument("--output", type=str, default="results/grading/")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.stage == "build-index":
        print("构建 FAISS 修正案例向量库")
        from src.ch3_sensitive_grading import RAGRetriever

    elif args.stage == "grade":
        print("运行分级实验（含 RAG + 自洽性校验）")
        from src.ch3_sensitive_grading import GradingPipeline

    elif args.stage == "ablation":
        print("运行消融实验（无画像/无RAG/无自洽性）")
        from src.ch3_sensitive_grading import GradingPipeline

    print(f"实验完成，结果保存至: {args.output}")


if __name__ == "__main__":
    main()
