#!/usr/bin/env python3
"""端到端检测实验脚本"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="端到端检测实验")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--output", type=str, default="results/e2e/")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("端到端泄露检测对照实验")
    print("=" * 60)
    print("\n比较方法:")
    print("  1. 静态规则（传统 DLP）")
    print("  2. DeBERTa 微调 + 静态规则")
    print("  3. 本文方法（动态画像驱动）")
    print("\n检测通道:")
    print("  - 邮件外发 (Email)")
    print("  - HTTP/云盘上传 (HTTP/Cloud)")
    print("  - USB/外设拷贝 (USB/Device)")
    print("  - 即时通讯文件传输 (IM)")

    print(f"\n结果保存至: {args.output}")


if __name__ == "__main__":
    main()
