# 部署指南

本文档涵盖两种部署模式：
- **单机 API 部署**：无 GPU 单机环境，通过 Qwen API 完成分级推理
- **C/S 分离部署**：GPU 服务器 + 多终端代理独立部署（论文图 4-3 架构）

---

## 目录

1. [环境准备](#1-环境准备)
2. [获取 API Key](#2-获取-api-key)
3. [配置 API Key](#3-配置-api-key)
4. [运行端到端 Demo](#4-运行端到端-demo)
5. [扫描单个文件](#5-扫描单个文件)
6. [目录持续监控](#6-目录持续监控)
7. [启动完整检测服务](#7-启动完整检测服务)
8. [C/S 分离部署](#8-cs-分离部署)
9. [验证 LLM 分类效果](#9-验证-llm-分类效果)
10. [常见问题排查](#10-常见问题排查)

---

## 1. 环境准备

### 1.1 系统要求

| 项目 | 最低要求 |
|------|----------|
| 操作系统 | Windows 10/11、Ubuntu 20.04+、macOS 12+ |
| Python | 3.10+ |
| 内存 | 8 GB（无需 GPU） |
| 网络 | 可访问 `dashscope.aliyuncs.com`（DashScope）或自建 API 端点 |

### 1.2 安装依赖

```bash
# 创建虚拟环境
conda create -n dlp python=3.10 -y
conda activate dlp

# 安装核心依赖（CPU 模式，跳过 GPU 相关包）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentence-transformers pyyaml numpy scipy pandas scikit-learn
pip install openai          # Qwen API 调用必需
pip install faiss-cpu       # 向量检索（CPU 版本）

# 如需完整安装（含 OCR、文档处理等）
pip install -r requirements.txt
```

> **关键依赖**：`openai>=1.0.0` — QwenAPIWrapper 使用 OpenAI Python SDK 调用 DashScope 兼容端点。

### 1.3 下载模型权重

API 模式下**仍需要**以下本地模型（用于感知层特征提取，非 LLM 推理）：

```bash
python scripts/download_models.py
```

下载内容：
- `sentence-transformers/all-MiniLM-L6-v2`（~90 MB）— SBERT 编码器
- `microsoft/layoutlmv3-base`（~500 MB）— 文档布局特征
- `models/unet_denoiser.pth`（~5 MB）— 扫描件去噪

不需要下载 Qwen2.5-7B 本地权重（~4.2 GB），LLM 推理完全通过 API 完成。

---

## 2. 获取 API Key

### 方案 A：阿里云 DashScope（推荐）

1. 注册阿里云账号：https://www.aliyun.com
2. 开通百炼平台：https://dashscope.console.aliyun.com
3. 在「API-KEY 管理」页面创建密钥
4. 复制 `sk-xxxxxxxxxxxxxxxxxxxxxxxx` 格式的 Key

可用模型及定价参考：

| 模型 | 输入价格 | 输出价格 | 适用场景 |
|------|---------|---------|----------|
| `qwen-turbo` | 0.3 元/百万 token | 0.6 元/百万 token | 快速测试、低成本 |
| `qwen-plus` | 0.8 元/百万 token | 2 元/百万 token | 推荐：效果与成本平衡 |
| `qwen-max` | 2 元/百万 token | 6 元/百万 token | 最高精度 |

> Demo 运行 5 个测试事件约消耗 2000-5000 token，费用不到 0.01 元。

### 方案 B：OpenAI 兼容 API（本地部署 / 第三方）

适用于使用 vLLM、Ollama、LiteLLM 等部署的本地模型：

```bash
# 示例：使用 Ollama 本地运行 Qwen
ollama pull qwen2.5:7b
ollama serve  # 默认监听 http://localhost:11434

# 示例：使用 vLLM 部署
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct --port 8000
```

---

## 3. 配置 API Key

### 方式一：环境变量（推荐，避免密钥写入文件）

```bash
# DashScope
export DASHSCOPE_API_KEY="sk-your-dashscope-key-here"

# Windows PowerShell
$env:DASHSCOPE_API_KEY="sk-your-dashscope-key-here"

# Windows CMD
set DASHSCOPE_API_KEY=sk-your-dashscope-key-here
```

### 方式二：修改配置文件

编辑 `configs/default_config.yaml`，修改 `models.qwen` 部分：

```yaml
models:
  qwen:
    enabled: true                     # <- 改为 true
    mode: "api"                       # <- 确保为 "api"
    # --- API 模式配置 ---
    api_key: "sk-your-key-here"       # <- 填入 API Key（或留空使用环境变量）
    api_model: "qwen-plus"            # DashScope: qwen-plus / qwen-turbo / qwen-max
    api_backend: "dashscope"          # "dashscope" 或 "openai"
    api_base_url: ""                  # DashScope 留空；OpenAI 兼容填地址
    temperature: 0.7
    top_p: 0.9
    max_new_tokens: 512
```

**OpenAI 兼容 API** 的配置示例（Ollama / vLLM）：

```yaml
models:
  qwen:
    enabled: true
    mode: "api"
    api_key: "not-needed"             # 本地服务通常不需要
    api_model: "qwen2.5:7b"           # Ollama 模型名
    api_backend: "openai"             # <- 改为 "openai"
    api_base_url: "http://localhost:11434/v1"  # <- 填入本地服务地址
    temperature: 0.7
    top_p: 0.9
```

---

## 4. 运行端到端 Demo

Demo 使用 5 个硬编码测试事件，覆盖典型场景：

| # | 场景 | 用户 | 操作 | 预期结果 |
|---|------|------|------|----------|
| 1 | 市场部员工邮件发送公开宣传册 | alice | send (Outlook) | PASS |
| 2 | 财务人员拷贝财报到 USB | bob | copy (Explorer) | ALERT |
| 3 | 运维人员非工作时间上传凭证文件 | charlie | upload (Chrome) | BLOCK |
| 4 | 市场部员工微信发送合同 | alice | send (WeChat) | ALERT |
| 5 | 运维人员本地查看配置文件 | charlie | read (Notepad++) | PASS |

### 运行命令

```bash
# 确保 API Key 已配置（环境变量或配置文件）
export DASHSCOPE_API_KEY="sk-your-key"

# 运行 Demo（CPU 模式，无需 GPU）
python main.py demo --config configs/default_config.yaml
```

### 预期输出

```
=================================================================
  Terminal Data Leakage Detection System — Demo
=================================================================

[init] Loading config: configs/default_config.yaml
  user vm1_alice          role=office_staff  dept=marketing
  user vm2_bob            role=finance       dept=finance
  user vm3_charlie        role=ops_admin     dept=infosec

[velociraptor] source: velociraptor-master/
  ...

-----------------------------------------------------------------
[detection] Processing 5 events through three-layer services
  PerceptionService -> ProfileMatchResult -> CognitionService
  CognitionService  -> GradingResult      -> ExecutionService
  ExecutionService  -> DetectionVerdict    -> EventBus broadcast
-----------------------------------------------------------------

  [PASS ] event 1: email/public/pass
         file  = C:/Users/alice/Documents/brochure_2026.pdf
         user  = vm1_alice  process = outlook.exe
         level = L1  risk = 0.123  conf = 0.85
         ...

  [ALERT] event 2: usb/finance/alert
         file  = C:/Users/bob/Documents/Q1_financial_report.xlsx
         user  = vm2_bob  process = explorer.exe
         level = L3  risk = 0.672  conf = 0.78
         vql  = SELECT TimeStamp, Filename, FullPath, Reason FROM ...
         ...

  [BLOCK] event 3: http/credential/block
         file  = C:/Admin/Keys/server_root_credentials.txt
         user  = vm3_charlie  process = chrome.exe
         level = L4  risk = 0.934  conf = 0.92
         vql  = SELECT System.TimeStamp AS TimeStamp, System.Proc...
         ...
```

每个事件的处理流程：

```
文件内容 ──→ [感知层] U-Net去噪 + 特征提取 + 聚类匹配
               ↓ ProfileMatchResult
         ──→ [认知层] Qwen API 分类分级 (L1~L4 敏感度标签)
               ↓ GradingResult
         ──→ [执行层] 风险决策 + VQL 规则生成
               ↓ DetectionVerdict (PASS / ALERT / BLOCK)
```

---

## 5. 扫描单个文件

对指定文件执行一次性检测：

```bash
# 基本用法
python main.py scan report.pdf --user alice

# 指定用户角色 + JSON 输出
python main.py scan C:/Users/bob/Documents/salary.xlsx \
    --user bob --role finance --json

# 使用自定义配置
python main.py scan contract.docx --user alice \
    --config configs/default_config.yaml
```

输出包含：
- 敏感度等级（L1-L4）
- 风险分数（0~1）
- 分级置信度
- 响应动作（PASS / ALERT / BLOCK）
- 生成的 VQL 检测规则（若为 ALERT/BLOCK）

---

## 6. 目录持续监控

监控指定目录的文件变动，实时检测：

```bash
# 监控目录（默认 5 秒轮询）
python main.py monitor --watch C:/Users/bob/Documents --user monitor_user

# 自定义轮询间隔
python main.py monitor --watch ./data/ --user monitor_user --interval 10
```

当监控目录中出现新文件或文件被修改时，系统自动：
1. 提取文件内容和特征
2. 调用 Qwen API 进行敏感度分类
3. 计算风险分数并输出检测结果

---

## 7. 启动完整检测服务

完整服务包含三层架构 + 事件总线 + 可选 Velociraptor 对接：

```bash
# CPU 模式启动（API 推理，无需 GPU）
python main.py serve --device cpu --config configs/default_config.yaml

# 启动并监控指定目录
python main.py serve --device cpu --watch C:/Users/bob/Documents

# 后台运行（Linux/macOS）
nohup python main.py serve --device cpu > logs/server.log 2>&1 &
```

### 初始化流程（8 步）

```
[1/8] Loading config
[2/8] Registering users (LDAP/config)
[3/8] PerceptionService ready   (ch2: IGW-Kmeans clustering)
[4/8] CognitionService ready    (ch3: Qwen API + RAG seed injection)
[5/8] ExecutionService ready    (ch4: adaptive thresholds)
[6/8] VelociraptorBridge        (offline — 单机模式可跳过)
[7/8] TerminalAgent started     (file watcher + process monitor)
[8/8] Event loop running        (Ctrl+C to stop)
```

---

## 8. C/S 分离部署

### 概述

论文图 4-3 描述了"GPU 服务器 + 多终端代理"的分离部署架构。代码通过 FastAPI 网络层实现，终端代理通过 HTTP/WebSocket 与服务端通信。

```
┌────────────────────────────────┐         HTTP/WS         ┌─────────────────────────┐
│   GPU 服务器                    │ ◄─────────────────────► │   终端 VM (Windows)      │
│   python main.py serve-api     │  POST /api/v1/events    │   python main.py agent   │
│                                │  ─────────────────────► │                          │
│   DLPServer (三层检测)          │  VerdictResponse        │   TerminalAgent          │
│   ├── PerceptionService (ch2)  │  ◄───────────────────── │   ├── FileSystemWatcher  │
│   ├── CognitionService  (ch3)  │  WS rule_push           │   ├── ProcessWatcher     │
│   ├── ExecutionService  (ch4)  │  ─────────────────────► │   ├── VQLExecutor        │
│   └── FastAPI (api_server.py)  │                         │   └── RemoteClient       │
└────────────────────────────────┘                         └─────────────────────────┘
```

### 8.1 安装额外依赖

```bash
pip install fastapi uvicorn[standard] pydantic websockets
```

> 这些依赖已包含在 `requirements.txt` 中。

### 8.2 服务端启动

```bash
# 基本启动（GPU 服务器，监听所有网卡）
python main.py serve-api --host 0.0.0.0 --port 8900 --device cuda

# CPU 模式（API 推理，无需 GPU）
python main.py serve-api --host 0.0.0.0 --port 8900 --device cpu

# 带 Token 认证（生产环境推荐）
python main.py serve-api --host 0.0.0.0 --port 8900 --api-token "your-secret-token"
```

启动后：
- Swagger UI 文档: `http://<server-ip>:8900/docs`
- 系统状态: `http://<server-ip>:8900/api/v1/status`

### 8.3 终端代理启动

```bash
# 基本启动
python main.py agent \
    --server http://192.168.1.100:8900 \
    --user vm2_bob \
    --watch C:/Users/bob/Documents

# 启用 WebSocket 实时规则推送
python main.py agent \
    --server http://192.168.1.100:8900 \
    --user vm2_bob \
    --watch C:/Users/bob/Documents \
    --ws

# 带 Token 认证 + 自定义 Agent ID
python main.py agent \
    --server http://192.168.1.100:8900 \
    --agent-id agent_vm2_bob \
    --user vm2_bob \
    --watch C:/Users/bob/Documents \
    --api-token "your-secret-token" \
    --interval 3
```

### 8.4 API 端点一览

| 方法 | 路径 | 功能 |
|------|------|------|
| POST | `/api/v1/agents/register` | Agent 注册 |
| POST | `/api/v1/agents/heartbeat` | 心跳保活 + 告知待拉规则数 |
| POST | `/api/v1/events` | 文件事件上报 → 三层检测 → 返回裁决 |
| GET | `/api/v1/rules/{agent_id}` | 拉取待下发 VQL 规则 |
| WebSocket | `/api/v1/ws/{agent_id}` | 双向通信（规则推送 + 事件上报） |
| GET | `/api/v1/status` | 系统状态 |
| GET | `/api/v1/agents` | 已注册 Agent 列表 |

### 8.5 通信流程

**事件上报流程**：
```
Agent (FileSystemWatcher 检测到文件变动)
  → RemoteClient.report_event()
    → POST /api/v1/events (file_content Base64 编码)
      → DLPServer.process_file_event()
        → PerceptionService → CognitionService → ExecutionService
      ← VerdictResponse (action, risk_score, sensitivity_level, vql_script)
  ← Agent 根据 response_action 执行本地动作
```

**规则下发流程（轮询模式）**：
```
Server (ExecutionService 生成 VQL 规则)
  → EventBus RULE_PUSHED → AgentManager 存入 pending_rules 队列
Agent (心跳线程定期执行)
  → GET /api/v1/rules/{agent_id}
  ← PendingRulesResponse (rules[])
  → VQLExecutor.load_rule() 加载到本地执行引擎
```

**规则下发流程（WebSocket 模式）**：
```
Server (ExecutionService 生成 VQL 规则)
  → EventBus RULE_PUSHED → AgentManager.push_rule()
    → WebSocket send_json({type: "rule_push", data: rule})
Agent (WebSocket 接收线程)
  → _handle_ws_message() → VQLExecutor.load_rule()
```

### 8.6 论文实验部署示例

复现论文第五章实验（1 台 GPU 服务器 + 3 台 Windows VM）：

```bash
# === 服务器 (Ubuntu 22.04, RTX 3090) ===
python main.py serve-api --host 0.0.0.0 --port 8900 --device cuda

# === VM-1: 行政人员 (Windows 10) ===
python main.py agent --server http://192.168.1.100:8900 \
    --user vm1_alice --watch C:/Users/alice/Documents

# === VM-2: 财务人员 (Windows 10) ===
python main.py agent --server http://192.168.1.100:8900 \
    --user vm2_bob --watch C:/Users/bob/Documents --ws

# === VM-3: 运维管理员 (Windows 10) ===
python main.py agent --server http://192.168.1.100:8900 \
    --user vm3_charlie --watch C:/Admin
```

### 8.7 与单进程模式的对比

| 维度 | `serve`（单进程） | `serve-api` + `agent`（C/S） |
|------|------------------|------------------------------|
| 部署方式 | Agent 与 Server 同进程 | Agent 和 Server 独立部署 |
| 通信方式 | Python 函数回调 | HTTP REST / WebSocket |
| 终端数 | 1 台 | 多台（每台运行一个 agent） |
| GPU 需求 | 服务器需要 GPU | 仅服务器需要 GPU，终端无需 |
| 适用场景 | 开发测试、Demo 演示 | 论文实验、生产部署 |
| API 文档 | 无 | 自动生成 Swagger UI (`/docs`) |

---

## 9. 验证 LLM 分类效果

### 9.1 快速验证 API 连通性

```python
# test_api.py — 单独验证 Qwen API 是否正常
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.qwen_api_wrapper import QwenAPIWrapper

wrapper = QwenAPIWrapper(
    api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
    model="qwen-plus",
    backend="dashscope",
)

# 测试单次调用
prompt = """请对以下文件进行敏感度分级。
文件名: Q1_financial_report.xlsx
内容摘要: Q1 Financial Statement - revenue 12.5M, net profit 3.2M
用户角色: finance, 部门: finance

请返回 JSON: {"level": "L1~L4", "category": "...", "reason": "...", "confidence": 0.0~1.0}"""

result = wrapper(prompt, seed=42)
print("API Response:")
print(result)

# 测试 MC-Dropout 模拟（不同 seed 应产生不同输出）
for seed in [42, 43, 44]:
    r = wrapper(prompt, seed=seed)
    print(f"\nseed={seed}: {r[:80]}...")
```

```bash
python test_api.py
```

### 9.2 端到端分类效果对比

在 Demo 输出中，关注以下字段验证 LLM 分类质量：

| 字段 | 含义 | 期望值 |
|------|------|--------|
| `level` | LLM 给出的原始敏感度等级 | 公开文件=L1, 内部=L2, 商密=L3, 核心=L4 |
| `conf` | 分级置信度 | >0.7 为正常, <0.7 触发等级升格 |
| `risk` | 最终风险分数 | 综合用户画像+环境+敏感度 |
| `vql` | 生成的 VQL 检测规则 | 仅 ALERT/BLOCK 时生成 |

### 9.3 API 模式 vs Stub 模式对比

```bash
# Stub 模式（qwen.enabled=false）— 随机分级，用于功能测试
# 修改 configs/default_config.yaml: qwen.enabled: false
python main.py demo

# API 模式（qwen.enabled=true）— 真实 LLM 分级
# 修改 configs/default_config.yaml: qwen.enabled: true
python main.py demo
```

API 模式下，LLM 会根据文件内容、用户角色、操作类型综合判断敏感度，分级结果应显著优于随机 Stub。

---

## 10. 常见问题排查

### Q1: `openai` 库未安装

```
ImportError: openai 库未安装。安装: pip install openai
```

**解决**：`pip install openai>=1.0.0`

### Q2: API Key 无效或未设置

```
[QwenAPI] DashScope attempt 1 failed: AuthenticationError(...)
[QwenAPI] All retries exhausted, returning fallback
```

**解决**：
- 检查环境变量：`echo $DASHSCOPE_API_KEY`（Linux）或 `echo %DASHSCOPE_API_KEY%`（Windows CMD）
- 检查 Key 是否以 `sk-` 开头
- 确认百炼平台账户已充值

> API 不可达时系统会自动降级为 hash 分级（`_fallback_response`），不会崩溃，但分类质量等同随机。

### Q3: 网络连接超时

```
[QwenAPI] DashScope attempt 1 failed: ConnectTimeout(...)
```

**解决**：
- 检查网络是否可访问 `dashscope.aliyuncs.com`
- 如有代理，设置 `HTTPS_PROXY` 环境变量
- 使用本地 OpenAI 兼容 API（Ollama/vLLM）替代

### Q4: Demo 结果全为 PASS

可能原因：
- `qwen.enabled` 仍为 `false`，使用了随机 Stub
- API 返回 fallback 响应（检查日志中是否有 `All retries exhausted`）

**解决**：确认 `qwen.enabled: true` 且 API 调用成功。

### Q5: 本地 SBERT / LayoutLMv3 模型下载失败

API 模式不需要 Qwen 本地权重，但感知层仍需要 SBERT 和 LayoutLMv3。

**解决**：
```bash
# 使用国内镜像
export HF_ENDPOINT=https://hf-mirror.com
python scripts/download_models.py
```

### Q6: 单机不部署 Velociraptor 是否影响检测？

不影响。Velociraptor 为可选组件（`velociraptor.enabled: false`），系统仍完成完整的三层检测流程：
- 感知层：用户画像匹配
- 认知层：Qwen API 文件分级
- 执行层：风险决策 + VQL 规则生成

VQL 规则会生成但不实际部署执行。Demo 中 `vql` 字段展示的是**可用于实际 Velociraptor 部署**的真实检测规则。

---

## 附录：配置速查

### 最小配置（单机 API Demo）

```yaml
# configs/default_config.yaml 关键字段
models:
  qwen:
    enabled: true
    mode: "api"
    api_key: ""              # 使用环境变量 DASHSCOPE_API_KEY
    api_model: "qwen-plus"
    api_backend: "dashscope"

velociraptor:
  enabled: false             # 单机无需 Velociraptor

hardware:
  gpu_ids: []                # 无 GPU
  num_workers: 4
```

### 环境变量汇总

| 变量名 | 用途 | 示例值 |
|--------|------|--------|
| `DASHSCOPE_API_KEY` | DashScope API 密钥 | `sk-xxxxxxxx` |
| `OPENAI_API_KEY` | OpenAI 兼容 API 密钥 | `sk-xxxxxxxx` |
| `HF_ENDPOINT` | HuggingFace 镜像地址 | `https://hf-mirror.com` |
| `HTTPS_PROXY` | 网络代理 | `http://proxy:8080` |
