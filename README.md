# 基于用户数据特征画像的终端数据泄露检测系统

## 1. 基本信息

| 字段 | 内容 |
|------|------|
| **项目名称** | dlp-profiling — Terminal Data Leakage Detection Based on User Data Feature Profiling |
| **对应论文** | 《基于用户数据特征画像的终端数据泄露检测方法研究》(硕士学位论文) |
| **作者/维护人** | 程庶伦 |
| **版本** | v2.0 |
| **编程语言** | Python 3.10 + Go 1.21 (Velociraptor) |
| **最后更新** | 2026-04 |

## 2. 功能说明

本项目实现了论文提出的"三层协同"终端数据泄露检测方法，包含三个核心模块：

| 模块 | 对应论文章节 | 功能 |
|------|-------------|------|
| **多模态用户数据特征画像构建** | 第 2 章 | U-Net 去噪 + LayoutLMv3 多模态编码，经双分支 CrossAttention 融合为多维特征，通过信息增益加权 K-means（IGW-Kmeans）在内容语义、行为模式、权限上下文三维空间构建动态画像 |
| **基于画像先验约束的敏感数据智能分级** | 第 3 章 | 画像序列化注入五层结构化提示，Top-p 注意力筛选 RAG 召回历史修正案例，MC-Dropout 不确定性量化 + 多数投票自洽性校验，输出 L1-L4 四级分级 |
| **语义驱动的自适应检测规则生成与执行** | 第 4 章 | 约束解码生成四元组 IR (Subject, Object, Action, Condition)，Z3 霍尔三元组形式验证，模板编译为 VQL 脚本，三维偏离度 R(a) 决策 PASS/ALERT/BLOCK |

### 核心结论对应

- **端到端检测**（论文 5.2 节）：F₁ = 0.942，误报率 4.4%，较 DeBERTa+静态规则方案 F₁ 提升 13.5%
- **画像构建**（论文 5.3 节）：IGW-Kmeans 轮廓系数 SC = 0.68，较标准 K-means 提升 61.9%
- **敏感分级**（论文 5.4 节）：分级 F₁ = 0.950，较 Presidio 提升 16.7%
- **规则生成**（论文 5.5 节）：语法合规率 99.4%，逻辑冲突率 0.3%

### 主要模块与文件对应

```
dlp-profiling/
├── main.py                            # 系统入口: serve/demo/scan/monitor/report 五种模式
├── configs/
│   ├── default_config.yaml            # 全局配置 (数据路径、模型、API)
│   ├── profiling_config.yaml          # 画像参数 
│   ├── grading_config.yaml            # 分级参数
│   └── detection_config.yaml          # 检测参数 
├── src/
│   ├── server.py                      # DLPServer: 三层服务编排 + EventBus 事件驱动
│   ├── api_server.py                  # FastAPI 服务端 API: C/S 分离部署的网络层
│   ├── api_client.py                  # 终端代理远程客户端: HTTP/WebSocket 通信
│   ├── pipeline.py                    # DLPPipeline: 简化版流水线
│   ├── event_bus.py                   # 事件总线: FILE_DETECTED → GRADING_DONE → RULE_PUSHED
│   ├── terminal_agent.py              # 终端代理: 文件监控 (watchdog) + 进程→通道识别
│   ├── velociraptor_bridge.py         # gRPC 桥接: Artifact 注册 + Client Monitoring 下发
│   ├── ch2_user_profiling/            # 第 2 章: 感知层
│   │   ├── unet_denoiser.py           #   算法 2.1: 4层U-Net去噪 (1.2M参数)
│   │   ├── document_parser.py         #   算法 2.2: LayoutLMv3 + OCR + SBERT 双路径解析
│   │   ├── fusion_encoder.py          #   算法 2.3: 双分支CrossAttention融合 → 896维
│   │   ├── feature_extraction.py      #   三维特征提取 (内容768d + 行为96d + 权限32d)
│   │   ├── igw_kmeans.py              #   算法 2.4: 信息增益加权K-means + 三步交替优化
│   │   ├── incremental_update.py      #   算法 2.5: EMA增量更新 + 马氏距离漂移检测
│   │   └── temporal_alignment.py      #   多源异构数据时空对齐
│   ├── ch3_sensitive_grading/         # 第 3 章: 认知层
│   │   ├── grading_pipeline.py        #   算法 3.1: 端到端分级管线
│   │   ├── prompt_builder.py          #   五层结构化提示模板构建
│   │   ├── rag_retriever.py           #   算法 3.2: FAISS + Top-p注意力筛选RAG
│   │   ├── self_consistency.py        #   算法 3.3: MC-Dropout(T=5) + 多数投票
│   │   └── asset_catalog.py           #   敏感资产目录维护
│   ├── ch4_rule_generation/           # 第 4 章: 执行层
│   │   ├── constrained_decoder.py     #   算法 4.1: FSM约束解码 → 四元组IR
│   │   ├── quadruple_ir.py            #   四元组中间表示 ⟨Subject,Object,Action,Condition⟩
│   │   ├── template_synthesizer.py    #   算法 4.2: Z3霍尔三元组验证 + 模板编译
│   │   ├── vql_compiler.py            #   VQL编译: watch_usn / watch_etw / diff+glob
│   │   ├── adaptive_decision.py       #   算法 4.3: 三维偏离度 R(a) + 动态白名单
│   │   └── grammar_correction.py      #   语法违例修正
│   ├── data_pipeline/                 # 数据集构建脚本 (6个)
│   ├── models/                        # 模型封装 (LayoutLMv3/SBERT/Qwen/U-Net)
│   ├── baselines/                     # 对比基线 (静态规则/DeBERTa/Presidio)
│   └── utils/                         # 工具 (日志/指标/数据加载)
├── scripts/                           # 实验脚本
│   ├── exp_01~11_*.py                 #   11个实验 → 论文表5.4~5.15
│   ├── train_unet.py                  #   U-Net训练 (α·MSE+(1-α)·(1-SSIM), α=0.7)
│   ├── train_deberta.py               #   DeBERTa基线训练 (lr=2e-5, batch=16, epoch=5)
│   └── run_all_experiments.py         #   运行全部实验
├── data/                              # 实验数据集 (详见 data/README.md)
├── results/                           # 实验结果表 (12张CSV)
├── figures/                           # 实验结果图 (5张PDF)
├── patent/                            # 论文相关专利材料
├── models/                            # 模型权重存放
├── tests/                             # 集成测试
├── velociraptor-master/               # Velociraptor源码 (VQL执行引擎参考)
└── docs/
    ├── DEPLOYMENT_GUIDE.md            # 部署指南 (单机API/C/S分离/DashScope/Ollama配置)
```

## 3. 环境说明

### 3.1 已验证环境

| 项目 | 规格 |
|------|------|
| **操作系统** | Ubuntu 22.04 LTS (服务器端) / Windows 10 (终端侧) |
| **CPU** | 32 vCPU (服务器), 8 vCPU (终端VM) |
| **内存** | 128 GB (服务器), 8 GB (终端VM) |
| **GPU** | NVIDIA RTX 3090 24GB (Qwen 4-bit 约4.2GB + LayoutLMv3 约1.8GB) |
| **存储** | >= 8 GB |

### 3.2 软件依赖

| 依赖 | 版本要求 | 用途 |
|------|---------|------|
| **Python** | >= 3.10 | 运行环境 |
| **PyTorch** | >= 2.1 | 深度学习框架 |
| **CUDA** | >= 12.1 | GPU 推理加速 (可选) |
| **transformers** | >= 4.36 | LayoutLMv3 / Qwen2.5-7B |
| **sentence-transformers** | >= 2.2 | Sentence-BERT 语义编码 |
| **faiss-gpu / faiss-cpu** | >= 1.7.4 | RAG 向量检索 |
| **z3-solver** | >= 4.12 | 霍尔三元组形式验证 |
| **fastapi + uvicorn** | >= 0.104 / >= 0.24 | C/S 分离部署 API 层 |
| **Go** | >= 1.21 | Velociraptor 编译与部署 (完整部署时需要) |
| **reportlab / python-docx / openpyxl** | 最新稳定版 | 文档生成与解析 |

完整依赖见 `requirements.txt`。

### 3.3 外部依赖

| 服务 | 说明 | 是否必需 |
|------|------|---------|
| **DashScope API** | 阿里云百炼 Qwen API, 用于敏感分级与规则解析 | 否 (无 API Key 时使用随机 stub 或本地模型) |
| **Velociraptor Server** | VQL 规则执行引擎, gRPC 对接 | 否 (未部署时 VQL 仅本地缓存) |

> **注意**: 即使没有 GPU、API Key 和 Velociraptor, 系统仍可在 CPU-only 模式下运行 Demo 和单文件扫描, 三层流水线逻辑完整可验证。

## 4. 使用步骤

### 4.1 Quick Start (最小可运行示例)

**目标**: 在一台机器上安装依赖并运行 Demo, 观察 5 条事件通过三层流水线输出 PASS/ALERT/BLOCK。

```bash
# 1. 进入项目目录
cd dlp-profiling

# 2. 创建环境并安装依赖
conda create -n dlp python=3.10 -y
conda activate dlp
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy pandas scikit-learn pyyaml tqdm matplotlib seaborn
pip install python-docx openpyxl reportlab PyPDF2 Pillow pymupdf z3-solver

# 3. 运行 Demo
python main.py demo

# 4. 预期输出 (成功标志):
#    ================================================================
#      Terminal Data Leakage Detection System — Demo
#    ================================================================
#    [init] Loading config: configs/default_config.yaml
#      user vm1_alice    role=office_staff   dept=marketing
#      user vm2_bob      role=finance        dept=finance
#      user vm3_charlie  role=ops_admin      dept=infosec
#
#    [detection] Processing 5 events through three-layer services
#
#      [PASS ] event 1: email/public/pass     level=L1  risk=0.182
#      [ALERT] event 2: usb/finance/alert     level=L3  risk=0.583  vql=SELECT...
#      [BLOCK] event 3: http/credential/block level=L4  risk=0.891  vql=SELECT...
#      [ALERT] event 4: im/contract/alert     level=L3  risk=0.556
#      [PASS ] event 5: local/config/pass     level=L2  risk=0.203
#
#    [summary] events=5  blocked=1  alerted=2  passed=2

# 5. 停止: 自动退出 (Demo 为一次性运行)
```

**运行成功的标志**:
- 输出 `Loading config:` 表示配置加载成功
- 输出 `Processing 5 events` 表示三层服务初始化完成
- 5 条事件均显示 `[PASS]`/`[ALERT]`/`[BLOCK]` 和 `risk=` 数值表示流水线正常工作
- event 2/3 带有 `vql=SELECT...` 表示 VQL 规则生成成功

**最常见的失败原因**:
- `ModuleNotFoundError: No module named 'xxx'`: 缺少依赖, 执行 `pip install xxx`
- `FileNotFoundError: configs/default_config.yaml`: 未在项目根目录执行, 检查 `cd` 路径
- `RuntimeError: CUDA not available`: 正常现象, CPU-only 模式自动回退, 不影响 Demo

**其他命令**:

```bash
# 扫描单个文件
python main.py scan report.pdf --user alice --action read --json

# 持续监控目录 (持久运行, Ctrl+C 停止)
python main.py monitor --watch ./data/custom --interval 5

# C/S 分离部署 (详见阶段 5b)
python main.py serve-api --port 8900                              # 服务端 (GPU)
python main.py agent --server http://ip:8900 --watch ./data       # 终端代理 (VM)

# 查看实验结果摘要
python main.py report
```

### 4.2 Full Reproduction (完整复现论文实验)

> **目标**: 复现论文第五章全部实验结果。
> **硬件**: 1 台 GPU 服务器 + 3 台终端虚拟机。

**总体流程**:

```
阶段 1  环境准备       ← 安装 GPU 版依赖 + 下载模型
阶段 2  数据验证       ← 确认已包含的数据集完整
阶段 3  模型训练       ← U-Net 去噪 + DeBERTa 基线 (可选)
阶段 4  实验运行       ← 11 个实验脚本
阶段 5  系统部署       ← 持久化服务 + Velociraptor 对接
```

#### 阶段 1: 环境准备

**目标**: 安装 GPU 版依赖 + 下载预训练模型

**前置条件**: GPU 服务器已安装 CUDA 12.1、conda

```bash
cd dlp-profiling

# 创建环境
conda create -n dlp python=3.10 -y
conda activate dlp

# GPU 版本安装
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 下载预训练模型（LayoutLMv3、Sentence-BERT、Qwen2.5-7B）
python scripts/download_models.py
```

**输出**: `models/` 目录下出现模型权重文件
**是否可跳过**: 否

#### 阶段 2: 数据验证

**目标**: 确认仓库中已包含的数据集完整

本仓库已包含全部实验数据（`data/unet_pairs/` 除外）。运行以下命令验证:

| 数据集 | 验证命令 | 预期结果 |
|--------|----------|----------|
| RVL-CDIP 子集 | `wc -l data/rvlcdip/labels.csv` | 3001 |
| Enron 邮件 PDF | `wc -l data/enron_pdf/labels.csv` | 701 |
| 自建文档 | `wc -l data/custom/labels.csv` | 1301 |
| 行为事件流 | `wc -l data/behavior_logs/vm*/events.jsonl` | ~85800 |
| 操作序列(异常) | `ls data/sequences/anomaly/ \| wc -l` | 380 |
| 操作序列(正常) | `ls data/sequences/normal/ \| wc -l` | 550 |
| 安全策略 | `wc -l data/policies/policies.jsonl` | 200 |

若需重建 U-Net 训练配对（2.4GB, 未包含）:

```bash
python src/data_pipeline/noise_augmentor.py \
    --clean-dir data/rvlcdip --output-dir data/unet_pairs --seed 42
# 输入: data/rvlcdip/ (3000 张 TIF)
# 输出: data/unet_pairs/ (clean/ 3000 张 + noisy/ 9000 张 + pairs.csv)
```

**是否可跳过**: 是（若文件数与上表一致）

#### 阶段 3: 模型训练 (可选)

**目标**: 训练 U-Net 去噪模型和 DeBERTa 基线

**前置条件**: 阶段 1 + 阶段 2 完成, GPU 可用

```bash
# 1. U-Net 去噪模型训练（RTX 3090）
python scripts/train_unet.py \
    --data-dir data/unet_pairs \
    --epochs 100 \
    --batch-size 16
# 输入: data/unet_pairs/
# 输出: models/unet_denoiser.pth
# 论文参数: 4层编码器-解码器, 1.2M参数, 损失=α·MSE+(1-α)·(1-SSIM), α=0.7

# 2. DeBERTa 基线训练（对比实验用）
python scripts/train_deberta.py \
    --data_dir data \
    --output_dir checkpoints/deberta
# 输入: data/ 下标注文档
# 输出: checkpoints/deberta/
# 论文参数: lr=2e-5, batch=16, epoch=5, AdamW, 10% linear warmup
```

**是否可跳过**: 是（若已有模型权重）

#### 阶段 4: 实验运行

**目标**: 复现论文表 5.4~5.15 全部实验结果

**前置条件**: 阶段 1 + 阶段 3 完成（或已有权重）

```bash
# 一键运行全部 11 个实验
python scripts/run_all_experiments.py --config configs/default_config.yaml

# 或逐个运行：
python scripts/exp_01_clustering_comparison.py    # 聚类质量对比          → 表 5.7
python scripts/exp_02_feature_weight_analysis.py  # 特征维度消融          → 表 5.8
python scripts/exp_03_k_sensitivity.py            # 聚类数K敏感性
python scripts/exp_04_incremental_update.py       # 增量更新验证
python scripts/exp_05_grading_baseline_comparison.py  # 分级多基线对比    → 表 5.9
python scripts/exp_06_format_grading.py           # 格式分类分级          → 表 5.10
python scripts/exp_07_context_ablation.py         # 上下文消融            → 表 5.11
python scripts/exp_08_consistency_sampling.py     # 自洽性采样次数        → 表 5.12
python scripts/exp_09_parameter_sensitivity.py    # 超参数γ/k分析         → 表 5.13, 5.14
python scripts/exp_10_rule_generation_quality.py  # 规则生成质量          → 表 5.15
python scripts/exp_11_e2e_detection.py            # 端到端检测对比        → 表 5.4, 5.5, 5.6
```

**输出路径**: `results/*.csv` — 对应论文各表格的结构化数据
**是否可跳过**: 是（`results/` 已包含从论文提取的全部实验数据）

#### 阶段 5: 系统部署 — 持久化运行

**目标**: 部署为持久化检测服务, 持续监控终端文件操作并实时执行检测

**前置条件**: 阶段 1 完成; Velociraptor Server 已部署 (可选)

##### 5a. 单机持久化（无 Velociraptor）

```bash
# 启动检测服务，监控指定目录
python main.py serve \
    --config configs/default_config.yaml \
    --device cuda \
    --watch C:/Users/bob/Documents \
    --interval 5

# 系统进入持久事件循环：
#   TerminalAgent 每 5 秒扫描目录变化
#   → 文件事件进入三层流水线
#   → 输出 PASS / ALERT / BLOCK
#   → VQL 规则本地缓存
#   Ctrl+C 或 SIGTERM 优雅退出
```

##### 5b. C/S 分离部署（论文图 4-3 架构）

论文图 4-3 描述的"服务端(GPU) + 多终端代理(VM)"独立部署模式，通过 FastAPI 网络层实现：

```
┌──────────────────────────────┐           HTTP/WS            ┌──────────────────────┐
│  GPU 服务器 (Ubuntu 22.04)    │  ◄─────────────────────────►  │  终端 VM-1 (Windows)  │
│  python main.py serve-api    │   POST /api/v1/events         │  python main.py agent │
│  ├─ FastAPI (api_server.py)  │   ──────────────────────────►  │  ├─ FileSystemWatcher │
│  │  ├─ /api/v1/events        │   裁决(VerdictResponse)        │  ├─ ProcessWatcher    │
│  │  ├─ /api/v1/rules/{id}    │   ◄──────────────────────────  │  ├─ VQLExecutor       │
│  │  └─ /api/v1/ws/{id}       │   WS 规则推送(rule_push)       │  └─ RemoteClient      │
│  ├─ PerceptionService (ch2)  │   ──────────────────────────►  │      (api_client.py)  │
│  ├─ CognitionService  (ch3)  │                                └──────────────────────┘
│  ├─ ExecutionService  (ch4)  │                                ┌──────────────────────┐
│  └─ VelociraptorBridge       │  ◄─────────────────────────►  │  终端 VM-2 (Windows)  │
└──────────────────────────────┘                                └──────────────────────┘
```

**服务端启动**（GPU 服务器）：

```bash
# 基本启动
python main.py serve-api --host 0.0.0.0 --port 8900 --device cuda

# 带 API Token 认证
python main.py serve-api --host 0.0.0.0 --port 8900 --api-token "your-secret-token"

# 启动后自动提供 Swagger 文档: http://<server-ip>:8900/docs
```

**终端代理启动**（各 Windows VM）：

```bash
# VM-1: 行政人员
python main.py agent \
    --server http://192.168.1.100:8900 \
    --user vm1_alice \
    --watch C:/Users/alice/Documents \
    --interval 5

# VM-2: 财务人员（启用 WebSocket 实时推送）
python main.py agent \
    --server http://192.168.1.100:8900 \
    --user vm2_bob \
    --watch C:/Users/bob/Documents \
    --ws

# VM-3: 运维管理员（带 Token 认证）
python main.py agent \
    --server http://192.168.1.100:8900 \
    --user vm3_charlie \
    --watch C:/Admin \
    --api-token "your-secret-token"
```

**API 端点**：

| 方法 | 路径 | 功能 |
|------|------|------|
| POST | `/api/v1/agents/register` | Agent 注册（含主机名、OS、监控目录） |
| POST | `/api/v1/agents/heartbeat` | Agent 心跳保活 |
| POST | `/api/v1/events` | 文件事件上报 → 三层检测 → 返回裁决 |
| GET | `/api/v1/rules/{agent_id}` | 拉取待下发 VQL 规则 |
| WebSocket | `/api/v1/ws/{agent_id}` | 双向通信: 规则实时推送 + 事件上报 |
| GET | `/api/v1/status` | 系统状态（用户数、Agent 数、事件计数） |
| GET | `/api/v1/agents` | 列出所有已注册 Agent 状态 |
| GET | `/docs` | Swagger UI 自动文档 |

**通信机制**：
- **事件上报**：Agent 检测到敏感文件操作 → Base64 编码文件内容 → POST 到服务端 → 服务端三层检测 → 返回 VerdictResponse（含 risk_score、sensitivity_level、response_action、vql_script）
- **规则下发（轮询模式）**：Agent 心跳时附带 GET 拉取待下发 VQL 规则 → 加载到本地 VQLExecutor
- **规则下发（WebSocket 模式）**：`--ws` 参数启用后，服务端生成规则时实时推送到 Agent，无需轮询

##### 5c. 论文实验环境说明（1 Server + 3 VM）

论文第五章实验基于以下架构设计：

```
┌──────────────────────────────────────────────────────┐
│  GPU 服务器（Ubuntu 22.04）                           │
│  python main.py serve --device cuda                   │
│  ├── PerceptionService  (LayoutLMv3 + IGW-Kmeans)    │
│  ├── CognitionService   (Qwen2.5-7B + RAG)          │
│  ├── ExecutionService   (约束解码 + Z3 + VQL)        │
│  └── VelociraptorBridge                              │
│       ├── 上行: subscribe_client_events() 接收事件    │
│       └── 下行: register_artifact() 推送规则          │
└──────────┬───────────────────────────────────────────┘
           │ gRPC (双向)
           │ ↑ 终端事件回传 (watch_monitoring)
           │ ↓ VQL 规则下发 (artifact_set + client_monitoring)
     ┌─────┼─────┐
     │     │     │
  ┌──▼──┐ ┌▼──┐ ┌▼──┐
  │VM-1 │ │VM-2│ │VM-3│  Windows 10 + Velociraptor Agent
  │行政 │ │财务│ │核心│  watch_usn / watch_etw / diff+glob
  └─────┘ └────┘ └────┘  → 持续执行 VQL 检测规则
```

**代码实现**：

- **三层检测流水线**：`DLPServer.process_file_event()` 依次通过感知层（画像匹配+增量更新）、认知层（LLM分级+RAG+自洽性校验）、执行层（风险决策+VQL生成），每次处理均触发画像增量更新。
- **规则下发（下行）**：`VelociraptorBridge.register_artifact()` + `set_client_monitoring()` 通过 gRPC 将 VQL 规则注册并下发到终端。
- **事件接收（上行）**：`VelociraptorBridge.subscribe_client_events()` 通过 `watch_monitoring()` VQL 持续流式接收终端回传的检测事件，后台线程自动将事件送入三层流水线处理（`main.py` Step 7b）。
- **本机监控（备用）**：`TerminalAgent` 以轮询方式监控本机 `--watch` 目录，适用于单机演示。
- **行为模拟**：论文实验中 3 台 VM 的行为事件由 `behavior_simulator.py` 基于 CERT r4.2 统计分布预生成（`data/behavior_logs/`）。

**如需对接真实 Velociraptor 环境**：

```bash
# 1. 服务器：启动 Velociraptor Server
velociraptor --config server.config.yaml frontend

# 2. 服务器：启动检测系统（自动连接 Velociraptor gRPC）
python main.py serve --device cuda --config configs/default_config.yaml

# 3. 各终端 VM：安装并注册 Velociraptor Agent
velociraptor --config client.config.yaml client -v

# 系统启动后自动完成：
#   [6/8] 连接 Velociraptor Server (gRPC)
#   [7/8] 启动 TerminalAgent (文件监控 + 进程识别)
#   [8/8] 进入主事件循环，持续等待文件事件
```

**Velociraptor 配置详见**: `docs/DEPLOYMENT_GUIDE.md`

**是否可跳过**: 可跳过（论文实验使用模拟事件流，不依赖实际 Velociraptor 部署）

### 4.3 运行模式说明

| 模式 | 命令 | 说明 | 持久化 |
|------|------|------|--------|
| `serve` | `python main.py serve --watch DIR` | 完整三层检测服务 + Velociraptor (单进程) | 是 |
| `serve-api` | `python main.py serve-api --port 8900` | C/S 模式服务端: FastAPI + 三层检测 (GPU 服务器) | 是 |
| `agent` | `python main.py agent --server URL --watch DIR` | C/S 模式终端代理: 远程上报 + 本地 VQL 执行 | 是 |
| `demo` | `python main.py demo` | 5 条预设事件演示完整流程 | 否 |
| `scan` | `python main.py scan FILE --user USER` | 扫描单个文件输出检测结果 | 否 |
| `monitor` | `python main.py monitor --watch DIR` | 轻量级目录监控（无 Velociraptor） | 是 |
| `report` | `python main.py report` | 查看实验结果摘要 | 否 |

### 4.3 数据说明

所有实验数据已包含在 `data/` 目录中 (详见 `data/README.md`):

| 目录 | 内容 | 规模 | 来源 |
|------|------|------|------|
| `data/rvlcdip/` | RVL-CDIP 文档图像子集 | 3,000 份 TIF | 公开数据集筛选 10 类 |
| `data/enron_pdf/` | Enron 邮件渲染 PDF | 700 份 PDF | 公开数据集关键词分级 |
| `data/custom/` | 自建多格式文档 | 1,300 份 (docx/xlsx/pdf) | 脚本生成 |
| `data/behavior_logs/` | 30 天终端行为事件流 | ~85,000 条 JSONL | CERT r4.2 分布校准模拟 |
| `data/sequences/` | 操作序列 | 930 条 (异常380+正常550) | 42 种窃取场景 + 4 通道 |
| `data/policies/` | 安全策略语料 | 200 条 JSONL | NIST/ATT&CK/Sigma/自建 |
| `data/unet_pairs/` | U-Net 训练配对 | 仅 README | 由脚本从 rvlcdip 重建 (2.4GB) |

> 本仓库不包含 `data/unet_pairs/` 的图像文件 (体积 2.4GB)。可由 `noise_augmentor.py` 从 `data/rvlcdip/` 一键重建，见目录内 README.md。

## 5. 结果对应说明

### 5.1 结果文件 → 论文图表对应

| 文件 (results/) | 论文表格 | 内容|
|-----------------|----------|------
| `tab_e2e_detection.csv` | 表 5.4 | 三种方法端到端检测对比 | 
| `tab_channel_detection.csv` | 表 5.5 | 四种泄露通道分通道检测效果 | 
| `tab_resilience.csv` | 表 5.6 | 角色迁移场景 6 个时段性能演化 | 
| `tab_cluster_compare.csv` | 表 5.7 | 4 种聚类方法质量对比 | 
| `tab_feature_ablation.csv` | 表 5.8 | 10 组特征维度消融实验 | 
| `tab_grading_compare.csv` | 表 5.9 | 3 种分级方法多基线对比 |
| `tab_format_grading.csv` | 表 5.10 | 按文档格式分类的分级 F1 | 
| `tab_context_ablation.csv` | 表 5.11 | 上下文增强消融实验 | 
| `tab_selfconsistency.csv` | 表 5.12 | 自洽性校验采样次数 T 分析 |
| `tab_gamma_tradeoff.csv` | 表 5.13 | 不确定性阈值 γ 权衡 | 
| `tab_param_k.csv` | 表 5.14 | RAG 召回案例数 k 敏感性 | 
| `tab_rule_quality.csv` | 表 5.15 | 3 种方法规则生成质量对比 | 

| 文件 (figures/) | 论文图号 | 内容 |
|-----------------|----------|------|
| `fig_channel_fpr.pdf` | 图 5.1 | 各通道检测误报率 |
| `fig_resilience.pdf` | 图 5.2 | 角色迁移性能演化 |
| `fig_ablation.pdf` | 图 5.3 | 上下文消融对比 |
| `fig_param_T.pdf` | 图 5.4 | 采样次数 T 敏感性 |
| `fig_param_k.pdf` | 图 5.5 | RAG 参数 k 敏感性 |

### 5.2 结果波动说明

- 检测率指标 (F₁/Precision) 受随机种子和 LLM 推理温度影响, 允许 ±2% 波动
- MC-Dropout 自洽性校验结果受 dropout mask 随机性影响, 一致性比率允许 ±3% 波动


## 6. 许可证

本代码仅用于学术研究目的。未经授权不得用于商业用途。
