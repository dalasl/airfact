# 数据集说明

## 1. 数据集概览

| 数据集 | 规模 | 格式 | 用途 | 获取方式 |
|--------|------|------|------|---------|
| RVL-CDIP 子集 | 3,000 份 | TIFF 扫描件 | 画像构建 + 分级 | 公开下载后脚本提取 |
| Enron 邮件 | 700 份 | 邮件渲染 PDF | 画像构建 + 分级 | 公开下载后脚本提取 |
| 自建文档 | 1,300 份 | Word/Excel/PDF/水印PDF | 画像构建 + 分级 | 脚本生成 |
| 终端行为事件 | ~85,000 条 | JSONL | 画像构建（行为模式维度） | 行为模拟器生成 |
| 操作序列 | ~930 条 | JSONL | 端到端检测验证 | 行为模拟器生成 |
| 安全策略语料 | 200 条 | JSONL | 规则生成质量评估 | 脚本整理 |
| U-Net 训练配对 | 9,000 对 | PNG | U-Net 去噪模型训练 | 噪声注入脚本生成 |
| CERT r4.2 | ~330 万事件 | CSV/日志 | 行为模式校准基准 | 需申请 |

## 2. 数据获取与处理

### 2.1 RVL-CDIP 子集
- **来源：** Harley et al., "Evaluation of Deep Convolutional Nets for Document Image Classification and Retrieval", ICDAR 2015
- **下载地址：** https://huggingface.co/datasets/rvl_cdip
- **处理脚本：** `src/data_pipeline/rvlcdip_processor.py`
- **处理说明：** 从原始 16 类中筛选 10 类企业相关文档（letter, memo, form, invoice, scientific_report, email, budget, scientific_publication, specification, file_folder），按敏感等级配额采样 3,000 份
- **敏感等级映射：** L1:750（科学报告/科学文献/文件夹封面）、L2:900（信件/邮件/备忘录）、L3:800（预算/发票）、L4:550（表格/规范说明）
- **许可：** 仅限学术研究用途

### 2.2 Enron Email Dataset
- **来源：** Klimt & Yang, "The Enron Corpus: A New Dataset for Email Classification Research", ECML 2004
- **下载地址：** https://huggingface.co/datasets/Pile-Enron_Emails
- **处理脚本：** `src/data_pipeline/enron_processor.py`
- **处理说明：** 从约 19 万封 JSONL 邮件中按敏感关键词分级筛选 700 封，使用 `reportlab` 渲染为 PDF 格式
- **敏感等级分布：** L1:180 / L2:220 / L3:180 / L4:120
- **许可：** 公开数据集

### 2.3 自建文档数据集
- **构建脚本：** `src/data_pipeline/custom_doc_generator.py`
- **构建方式：** 使用 `python-docx`、`openpyxl`、`reportlab` 等库，基于分类别内容段落池程序化生成
- **格式分布：**
  - Word (.docx): 500 份 — 行政通知、内部备忘、财务文档、合同协议等
  - Excel (.xlsx): 300 份 — 含格式化表头、分类别数据行和备注
  - PDF: 300 份 — 中文字体渲染，含公开报告、项目文档、技术方案等
  - 水印 PDF: 200 份 — 在原生 PDF 上叠加 CONFIDENTIAL 水印（透明度 0.2-0.5）
- **敏感等级分布：** L1:280 / L2:380 / L3:370 / L4:270
- **合规声明：** 所有内容为模拟生成，不包含真实敏感信息

### 2.4 终端行为事件与操作序列
- **构建脚本：** `src/data_pipeline/behavior_simulator.py`
- **设计依据：** CERT Insider Threat r4.2 同岗位统计分布校准
- **行为事件流：** 30 天 × 3 台 VM 差异化模拟，共 ~85,000 条原始事件
  - VM-1（行政办公）: ~28,000 条，第 8 天角色迁移至核心业务
  - VM-2（财务分析）: ~28,000 条，L2-L3 为主
  - VM-3（核心业务）: ~30,000 条，L3-L4 为主
- **操作序列：** 从事件流按泄露场景聚合 ~930 条操作链
  - 异常序列 ~380 条（42 个数据窃取场景，三级难度）
  - 正常序列 ~550 条
  - 覆盖邮件(100+150)、HTTP/云盘(95+140)、USB(95+140)、即时通讯(90+120) 四通道

### 2.5 安全策略语料
- **构建脚本：** `src/data_pipeline/policy_collector.py`
- **来源组成：**
  - NIST SP 800-53 Rev.5: 71 条
  - MITRE ATT&CK: 40 条
  - Sigma Rules 社区: 30 条
  - CIS Controls / ISO 27001 / 国内法规: 30 条
  - 自建终端策略: 29 条
- **复杂度分布：** 简单 70 / 中等 80 / 复杂 50
- **许可：** NIST/MITRE 内容为公共领域，Sigma Rules 遵循 DRL 1.1 许可

### 2.6 U-Net 去噪训练配对
- **构建脚本：** `src/data_pipeline/noise_augmentor.py`
- **来源：** 基于 RVL-CDIP 3,000 张干净图像
- **噪声类型：** gaussian / salt_pepper / jpeg（每张 × 3 = 9,000 配对）
- **用途：** 训练 U-Net 去噪模型，抑制扫描件噪声与水印干扰

### 2.7 CERT Insider Threat Dataset r4.2
- **来源：** CMU Software Engineering Institute, DARPA ADAMS 项目
- **下载地址：** https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247
- **访问要求：** 需注册并同意使用协议
- **规模：** 1,000 名模拟员工，501 个工作日，~330 万多源行为事件
- **本文使用方式：** 仅提取统计分布特征用于行为模拟器的操作分布校准，不直接作为实验数据

## 3. 敏感性标注说明

### 四级分类体系

| 等级 | 名称 | 定义 | 示例 |
|------|------|------|------|
| L1 | 公开级 | 可对外公开的信息 | 公开通知、行政公告、产品手册 |
| L2 | 内部级 | 仅限组织内部使用 | 内部通知、会议纪要、日常报表 |
| L3 | 机密级 | 严格限制访问范围 | 合同文本、技术方案、财务数据 |
| L4 | 绝密级 | 最高级别保护 | 战略决策、核心技术、关键资产 |

### 标注方式
- **RVL-CDIP：** 按文档类别到敏感等级的业务语义映射自动标注
- **Enron：** 按邮件内容敏感关键词自动分级
- **自建文档：** 按生成时预设的类别和敏感等级自动标注

## 4. 数据版权与合规

- **RVL-CDIP / Enron：** 公开学术数据集，仅用于学术研究
- **CERT r4.2：** 模拟数据，需遵守 CMU SEI 使用协议
- **自建数据集：** 程序化生成，不包含真实个人信息或企业数据
- **安全策略语料：** NIST/MITRE 为公共领域；Sigma Rules 遵循开源许可

**所有数据仅用于学术论文实验验证，不可用于商业目的。**

## 5. 目录结构

```
data/
├── rvlcdip/                # RVL-CDIP 3,000 份 TIF + labels.csv
├── enron_pdf/              # Enron 700 份 PDF + labels.csv
├── custom/                 # 自建 1,300 份 (docx/xlsx/pdf/wm.pdf) + labels.csv
├── behavior_logs/          # 行为事件流 ~85,000 条
│   ├── vm1/events.jsonl
│   ├── vm2/events.jsonl
│   └── vm3/events.jsonl
├── sequences/              # 操作序列 ~930 条
│   ├── anomaly/            #   异常 380 条
│   ├── normal/             #   正常 550 条
│   └── summary.json
├── policies/               # 安全策略 200 条
│   ├── policies.jsonl
│   └── summary.json
├── unet_pairs/             # U-Net 训练配对 9,000 对
│   ├── clean/              #   干净图像 3,000 张
│   ├── noisy/              #   噪声图像 9,000 张
│   └── pairs.csv
└── README.md               # 本文件
```

## 6. 一键构建

```bash
# 1. RVL-CDIP 子集提取（需先下载源数据集到 D:/datasets/rvl-cdip.tar/）
python src/data_pipeline/rvlcdip_processor.py

# 2. Enron 邮件提取与 PDF 渲染（需先下载源数据集到 D:/datasets/Pile-Enron_Emails/）
python src/data_pipeline/enron_processor.py

# 3. 自建文档生成
python src/data_pipeline/custom_doc_generator.py

# 4. 行为模拟器（生成事件流 + 操作序列）
python src/data_pipeline/behavior_simulator.py

# 5. 安全策略语料整理
python src/data_pipeline/policy_collector.py

# 6. U-Net 噪声配对（需先完成步骤 1）
python src/data_pipeline/noise_augmentor.py --clean-dir data/rvlcdip --output-dir data/unet_pairs
```
