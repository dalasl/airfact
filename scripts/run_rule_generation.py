#!/usr/bin/env python3
"""
检测规则生成与执行实验脚本

支持阶段：generate | execute
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="检测规则生成与执行实验")
    parser.add_argument("--stage", type=str, required=True,
                        choices=["generate", "execute"])
    parser.add_argument("--policies", type=str, default="data/raw/security_policies/")
    parser.add_argument("--rules", type=str, default="results/rule_generation/")
    parser.add_argument("--profiles", type=str, default="results/profiling/")
    parser.add_argument("--events", type=str, default="data/raw/behavior_logs/")
    parser.add_argument("--config", type=str, default="configs/detection_config.yaml")
    parser.add_argument("--output", type=str, default="results/rule_generation/")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.stage == "generate":
        print("运行规则生成质量评估")
        from src.ch4_rule_generation import TemplateSynthesizer, VQLCompiler

    elif args.stage == "execute":
        print("运行场景自适应执行实验")
        from src.ch4_rule_generation import AdaptiveDecisionEngine

    print(f"实验完成，结果保存至: {args.output}")


if __name__ == "__main__":
    main()
