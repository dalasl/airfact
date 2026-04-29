# 模型权重说明

## 所需模型

| 模型 | 用途 | 大小 | 来源 | 下载命令 |
|------|------|------|------|---------|
| U-Net Denoiser | 文档抗噪 | ~5MB | 自训练 | 见下方说明 |
| LayoutLMv3-base | 多模态文档理解 | ~500MB | HuggingFace | `python scripts/download_models.py` |
| Sentence-BERT | 语义向量编码 | ~90MB | HuggingFace | `python scripts/download_models.py` |
| Qwen2.5-7B-Instruct | LLM 推理骨干 | ~4.2GB (4-bit) | HuggingFace/ModelScope | `python scripts/download_models.py` |

## 获取方式

### 自动下载（推荐）

```bash
# 需要安装 huggingface_hub
pip install huggingface_hub

# 一键下载所有模型
python scripts/download_models.py
```

### 手动下载

#### LayoutLMv3
```bash
# HuggingFace
git lfs install
git clone https://huggingface.co/microsoft/layoutlmv3-base models/layoutlmv3/
```

#### Sentence-BERT
```bash
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 models/sentence_bert/
```

#### Qwen2.5-7B-Instruct (4-bit 量化)
```bash
# HuggingFace
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct models/qwen/

# 或 ModelScope（国内推荐）
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir='models/qwen/')"
```

#### U-Net Denoiser
自训练模型。使用论文中描述的 5000 训练对 + 500 验证对进行训练。
训练脚本：`scripts/run_profiling.py --stage denoise`
预训练权重文件（如有）放置于 `models/unet_denoiser.pth`

## FAISS 向量索引

RAG 模块使用的 FAISS IVF-Flat 索引在运行时自动构建，无需预下载。
构建命令：`python scripts/run_grading.py --stage build-index`

索引配置：
- 索引类型：IVF-Flat
- 向量维度：768
- 分区数 (n_list)：64
- 探测数 (n_probe)：8

## 硬件要求

| 模型 | GPU 显存 | 推理延迟 |
|------|---------|---------|
| U-Net | <100MB | ~15ms/page |
| LayoutLMv3 | ~1.8GB | ~320ms/page |
| Sentence-BERT | ~200MB | ~5ms/sentence |
| Qwen2.5-7B (4-bit) | ~4.2GB | ~1.2s/inference |
| **总计** | **~6.3GB** | — |

推荐显卡：NVIDIA RTX 3090 24GB 或以上

## 模型版本说明

本实验使用的模型版本为 2026 年 4 月时的最新稳定版。
如模型仓库有更新，建议使用相近版本以确保可复现性。
