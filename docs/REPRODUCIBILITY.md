## 1. 实验环境

### 硬件配置
| 设备 | 配置 |
|------|------|
| CPU | Intel Xeon Gold 6330 ×2 (56核112线程) |
| 内存 | 256GB DDR4 ECC |
| GPU | NVIDIA RTX 3090 24GB ×2 |
| 存储 | 2TB NVMe SSD |

### 软件环境
| 组件 | 版本 |
|------|------|
| 操作系统 | Ubuntu 22.04 LTS |
| Python | 3.10.12 |
| PyTorch | 2.1.0+cu121 |
| CUDA | 12.1 |
| cuDNN | 8.9 |
| Transformers | 4.36.2 |
| FAISS | 1.7.4 (GPU) |
| Velociraptor | 0.7.1 |

### 终端测试环境
| VM | 操作系统 | 配置 |
|----|---------|------|
| VM1 | Windows 11 Pro | 8 vCPU, 16GB RAM |
| VM2 | Windows 11 Pro | 4 vCPU, 8GB RAM |
| VM3 | Windows 10 Pro | 4 vCPU, 8GB RAM |

## 2. 随机性控制

- 全局随机种子：42
- IGW-Kmeans：10 次独立运行，报告 mean ± std
- 分级实验：5 折分层交叉验证
- MC-Dropout：每次推理使用递增种子 (0, 1, ..., T-1)
- 规则生成：每条策略生成 5 次

## 3. 统计检验

| 实验 | 检验方法 | 显著性水平 |
|------|---------|-----------|
| 聚类质量对比 | Wilcoxon 符号秩检验 | α = 0.05 |
| 分级方法对比 | McNemar 检验 | p < 0.01 |
| 标注一致性 | Fleiss' Kappa | κ = 0.83 |

## 4. 实验复现步骤

```bash
# 步骤 1: 环境搭建
conda env create -f environment.yml
conda activate dlp-profiling

# 步骤 2: 下载模型
python scripts/download_models.py

# 步骤 3: 准备数据
python scripts/prepare_datasets.py

# 步骤 4: 运行全部实验
python scripts/run_all_experiments.py --config configs/default_config.yaml

# 步骤 5: 检查结果
ls results/
```

## 5. 已知限制

1. **Qwen2.5-7B 版本依赖：** 不同量化版本可能产生略有差异的输出
2. **FAISS 索引非确定性：** IVF 聚类过程存在微小随机性，可能导致检索结果有细微差异
3. **CERT 数据集版本：** 必须使用 r4.2 版本，其他版本的场景编号和标注可能不同
4. **终端代理版本：** Velociraptor 0.7.1 的 VQL 语法与更高版本可能存在兼容性差异
