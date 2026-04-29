#!/usr/bin/env python3
"""模型下载脚本"""

import os

MODELS = {
    "LayoutLMv3": {
        "source": "HuggingFace",
        "repo": "microsoft/layoutlmv3-base",
        "size": "~500MB",
    },
    "Sentence-BERT": {
        "source": "HuggingFace",
        "repo": "sentence-transformers/all-MiniLM-L6-v2",
        "size": "~90MB",
    },
    "Qwen2.5-7B-Instruct": {
        "source": "HuggingFace / ModelScope",
        "repo": "Qwen/Qwen2.5-7B-Instruct",
        "size": "~4.2GB (4-bit quantized)",
        "quantization": "GPTQ 4-bit",
    },
}


def download_all():
    print("=" * 50)
    print("模型下载")
    print("=" * 50)

    for name, info in MODELS.items():
        print(f"\n--- {name} ---")
        print(f"  来源: {info['source']}")
        print(f"  仓库: {info['repo']}")
        print(f"  大小: {info['size']}")

        try:
            from huggingface_hub import snapshot_download
            save_dir = os.path.join("models", name.lower().replace(" ", "_").replace("-", "_"))
            os.makedirs(save_dir, exist_ok=True)
            print(f"  正在下载到 {save_dir}...")
            snapshot_download(repo_id=info["repo"], local_dir=save_dir)
            print(f"  ✓ 下载完成")
        except ImportError:
            print(f"  ✗ 需要安装 huggingface_hub: pip install huggingface_hub")
        except Exception as e:
            print(f"  ✗ 下载失败: {e}")
            print(f"  请手动下载: https://huggingface.co/{info['repo']}")


if __name__ == "__main__":
    download_all()
