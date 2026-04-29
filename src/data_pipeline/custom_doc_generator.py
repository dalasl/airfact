"""
自建多格式文档数据集生成器

使用模板+随机内容生成 1300 份企业办公文档：
- 500 份 Word (.docx)，含页眉/Logo/不同样式
- 300 份 Excel (.xlsx)，含格式化/隐藏列
- 300 份原生 PDF
- 200 份水印 PDF（透明度 0.2-0.5）

各文档预标注 L1-L4 敏感等级。
输出: data/custom/ + labels.csv
"""

import csv
import os
import random
from typing import Dict, List, Optional, Tuple

# ---- 文档类别模板 ----

DOCUMENT_TEMPLATES: Dict[str, List[Dict]] = {
    "L1": [
        {
            "category": "行政通知",
            "titles": [
                "关于{month}月考勤安排的通知",
                "办公区域管理规定",
                "{dept}部门培训计划",
                "员工手册更新说明",
                "假期安排通知",
            ],
            "body_template": (
                "各位同事：\n\n根据公司管理规定，现将{topic}相关事项通知如下：\n\n"
                "一、{section1}\n{detail1}\n\n"
                "二、{section2}\n{detail2}\n\n"
                "三、{section3}\n{detail3}\n\n"
                "特此通知。\n\n{dept}部\n{date}"
            ),
        },
        {
            "category": "公开报告",
            "titles": [
                "{year}年度企业社会责任报告",
                "产品使用手册 v{version}",
                "技术白皮书：{topic}",
            ],
            "body_template": (
                "摘要\n\n{detail1}\n\n"
                "1. {section1}\n{detail2}\n\n"
                "2. {section2}\n{detail3}\n\n"
                "3. {section3}\n{detail4}"
            ),
        },
    ],
    "L2": [
        {
            "category": "内部备忘",
            "titles": [
                "关于{project}项目进度的备忘录",
                "{dept}部门周报 - 第{week}周",
                "会议纪要：{topic}讨论会",
                "内部沟通：{topic}事项",
            ],
            "body_template": (
                "备忘录\n\n发件人：{sender}\n收件人：{receiver}\n日期：{date}\n"
                "主题：{subject}\n\n"
                "一、{section1}\n{detail1}\n\n"
                "二、{section2}\n{detail2}\n\n"
                "三、{section3}\n{detail3}\n\n"
                "下次会议：{next_meeting}"
            ),
        },
        {
            "category": "项目文档",
            "titles": [
                "{project}需求规格说明书",
                "{project}测试报告",
                "系统架构设计文档 - {module}",
            ],
            "body_template": (
                "1. 文档信息\n版本：{version}\n作者：{author}\n日期：{date}\n\n"
                "2. {section1}\n{detail1}\n\n"
                "3. {section2}\n{detail2}\n\n"
                "4. {section3}\n{detail3}"
            ),
        },
    ],
    "L3": [
        {
            "category": "财务文档",
            "titles": [
                "{year}年{quarter}财务报表",
                "{dept}部门预算审批单",
                "采购合同：{vendor}",
                "费用报销审批：{item}",
            ],
            "body_template": (
                "机密文件\n\n"
                "文档编号：{doc_no}\n密级：机密\n\n"
                "一、{section1}\n{detail1}\n\n"
                "二、{section2}\n{detail2}\n\n"
                "三、{section3}\n{detail3}\n\n"
                "审批人：{approver}\n日期：{date}"
            ),
        },
        {
            "category": "合同协议",
            "titles": [
                "保密协议 - {party}",
                "技术合作框架协议",
                "数据处理协议 - {vendor}",
            ],
            "body_template": (
                "合同编号：{contract_no}\n\n"
                "甲方：{party_a}\n乙方：{party_b}\n\n"
                "第一条 {section1}\n{detail1}\n\n"
                "第二条 {section2}\n{detail2}\n\n"
                "第三条 {section3}\n{detail3}\n\n"
                "第四条 {section4}\n{detail4}\n\n"
                "签署日期：{date}"
            ),
        },
    ],
    "L4": [
        {
            "category": "核心技术",
            "titles": [
                "核心算法设计文档：{algorithm}",
                "系统安全架构设计 v{version}",
                "密钥管理方案",
                "漏洞分析报告：{vuln_id}",
            ],
            "body_template": (
                "绝密文件\n\n"
                "文档编号：{doc_no}\n密级：绝密\n访问控制：仅限{access_group}\n\n"
                "1. {section1}\n{detail1}\n\n"
                "2. {section2}\n{detail2}\n\n"
                "3. {section3}\n{detail3}\n\n"
                "4. {section4}\n{detail4}"
            ),
        },
        {
            "category": "敏感人事",
            "titles": [
                "高管薪酬方案 - {year}",
                "组织架构调整方案（草案）",
                "裁员计划：{dept}部门",
            ],
            "body_template": (
                "绝密 - 仅限董事会成员\n\n"
                "方案名称：{plan_name}\n起草人：{author}\n日期：{date}\n\n"
                "一、{section1}\n{detail1}\n\n"
                "二、{section2}\n{detail2}\n\n"
                "三、{section3}\n{detail3}\n\n"
                "四、{section4}\n{detail4}"
            ),
        },
    ],
}

# ---- 随机词库 ----

_DEPTS = ["技术", "产品", "市场", "财务", "人力", "法务", "运维", "安全", "数据"]
_PROJECTS = ["Phoenix", "Atlas", "Titan", "Nova", "Omega", "Apex", "Nebula"]
_NAMES = ["张三", "李四", "王五", "赵六", "刘七", "陈八", "周九", "吴十"]
_VENDORS = ["华为", "阿里云", "腾讯", "字节跳动", "美团", "京东", "百度"]
_MODULES = ["认证模块", "数据层", "API网关", "消息队列", "权限中心", "日志平台"]
_ALGORITHMS = ["推荐引擎", "风控模型", "加密方案", "索引算法", "异常检测模型", "特征召回链路"]
_SYSTEMS = ["客户画像平台", "预算管理系统", "数据交换总线", "研发协同系统", "权限治理平台", "日志分析平台"]

_CATEGORY_TOPICS: Dict[str, List[str]] = {
    "行政通知": ["考勤排班", "办公环境优化", "门禁权限调整", "培训组织", "节假日值班"],
    "公开报告": ["绿色供应链", "数据安全治理", "产品服务能力", "客户成功体系", "隐私合规实践"],
    "内部备忘": ["资源协调", "风险复盘", "系统升级", "需求收敛", "预算使用"],
    "项目文档": ["接口联调", "容量扩展", "灰度发布", "多活容灾", "权限治理"],
    "财务文档": ["预算执行", "采购付款", "费用核销", "成本归集", "合同结算"],
    "合同协议": ["数据处理", "技术合作", "交付验收", "保密义务", "服务等级"],
    "核心技术": ["模型训练", "密钥轮换", "访问审计", "流量调度", "漏洞修复"],
    "敏感人事": ["组织优化", "岗位调整", "薪酬盘点", "继任安排", "裁撤评估"],
}

_CATEGORY_SECTION_TITLES: Dict[str, List[str]] = {
    "行政通知": ["执行范围", "时间安排", "流程说明", "资源准备", "检查机制", "例外处理"],
    "公开报告": ["治理进展", "服务能力", "合规实践", "客户价值", "运营成效", "后续规划"],
    "内部备忘": ["背景说明", "关键进展", "问题归因", "协调事项", "行动安排", "升级建议"],
    "项目文档": ["建设目标", "方案设计", "测试与发布", "依赖与约束", "风险控制", "验收标准"],
    "财务文档": ["事项概览", "资金测算", "审批建议", "预算执行", "付款安排", "风险提示"],
    "合同协议": ["合作范围", "权利义务", "保密安排", "违约责任", "争议处理", "交付验收"],
    "核心技术": ["技术目标", "核心实现", "安全控制", "部署策略", "性能指标", "审计要求"],
    "敏感人事": ["背景分析", "方案安排", "影响评估", "实施计划", "沟通机制", "保留策略"],
}

_CATEGORY_SLOT_COUNTS: Dict[str, Tuple[int, int]] = {
    "行政通知": (3, 3),
    "公开报告": (3, 4),
    "内部备忘": (3, 3),
    "项目文档": (3, 3),
    "财务文档": (3, 3),
    "合同协议": (4, 4),
    "核心技术": (4, 4),
    "敏感人事": (4, 4),
}


def _cross_join(prefixes: List[str], suffixes: List[str]) -> List[str]:
    return [f"{prefix}{suffix}" for prefix in prefixes for suffix in suffixes]


_CATEGORY_PARAGRAPHS: Dict[str, List[str]] = {
    "行政通知": _cross_join(
        [
            "自{date}起，{dept}部门统一执行新版考勤和补卡规则，所有外出、加班与调休申请需在OA系统内完成登记，",
            "{office_location}{floor}办公区本周启动工位和门禁权限同步调整，涉及设备搬迁、访客登记和座位编码更新的事项须按清单处理，",
            "{month}月行政培训将覆盖信息安全、消防演练和办公制度宣导三项内容，参训名单由各部门接口人汇总后统一提交，",
            "节假日前后办公资源申请量明显上升，为避免会议室、车辆和访客权限重复占用，所有预订动作均需提前录入系统，",
            "综合管理部将对办公用品申领、纸质单据流转和现场值班记录开展专项抽查，相关流程从本周起全部切换到新模板，",
        ],
        [
            "综合管理部将在每周五导出异常清单，对未按时提交说明的记录进行二次核验并在运营例会上通报。",
            "各部门接口人需在次日12:00前确认人员名单与门禁状态，避免因信息滞后影响办公秩序和审批节奏。",
            "如遇系统故障，员工须先提交纸质登记表，恢复后24小时内补录，确保台账、审批记录和审计留痕保持一致。",
        ],
    ),
    "公开报告": _cross_join(
        [
            "本报告围绕{year}年度经营与治理实践编制，覆盖客户服务、供应链协同、绿色办公与员工发展等核心议题，",
            "白皮书根据{review_date}前归档的项目台账、客户回访与合规审查结果整理，重点说明{topic}相关的阶段性进展，",
            "产品使用手册结合售前、实施和运维团队的对外交付经验，总结了部署边界、能力范围和常见风险提示，",
            "对外披露材料由品牌、法务和合规团队联合审阅，所有经营指标、案例描述和流程口径均以正式归档版本为准，",
            "公司在客户成功体系中新增了统一披露模板，对服务承诺、问题升级路径和数据处理边界进行持续标准化治理，",
        ],
        [
            "材料明确区分已落地措施与规划性安排，避免外部读者将阶段性目标误解为已承诺交付的最终结果。",
            "文档同时给出指标释义、样本范围和审校责任人，确保市场宣传、客户沟通与监管留痕在表述上保持一致。",
            "后续版本将根据客户反馈和监管变化及时修订术语与案例，使公开资料既可读又具备可审计和可追溯属性。",
        ],
    ),
    "内部备忘": _cross_join(
        [
            "{project}项目进入第{week}周后，需求冻结与开发排期出现偏差，主要受外部接口交付延迟、测试环境容量不足和审批窗口重叠影响，",
            "{dept}部门过去两周累计收到{issue_count}项与{topic}相关的协调请求，其中资源调拨、采购到货和供应商排期问题最为集中，",
            "项目组对现网告警、客户反馈与内部工单进行复盘后发现，同类问题反复出现的根因集中在责任边界不清和信息同步滞后，",
            "为支撑{system_name}下阶段灰度发布，当前必须在需求变更、联调节奏和培训安排三方面尽快形成统一执行口径，",
            "财务、采购和业务侧同步反馈，本月与{vendor}相关的合同补充条款尚未全部回签，已对测试设备到货与付款节点产生影响，",
        ],
        [
            "建议所有新增事项统一进入变更池评审，由产品、研发、测试和业务共同确认优先级后再进入排期和资源分配流程。",
            "各责任人需在日报中补充阻塞原因、预计恢复日期和所需支持资源，避免周报只描述现象而无法支持管理层决策。",
            "涉及日志导出、权限调整和个人信息处理的临时方案不得先上后补，必须同步形成审批记录、风险说明和回退计划。",
        ],
    ),
    "项目文档": _cross_join(
        [
            "{project}项目定位为支撑{system_name}的统一能力底座，本阶段目标是在不改变上层业务入口的前提下完成关键链路重构，",
            "当前版本聚焦{module}与周边系统的解耦，要求在{go_live_date}前实现配置可灰度、接口可回滚和日志可追溯，",
            "本次设计延续上一阶段的微服务拆分原则，将高频调用能力下沉到共享组件，并以统一配置中心管理环境差异，",
            "文档适用范围覆盖研发、测试、运维与业务接口团队，重点解释模块边界、输入输出约束与非功能指标，",
            "系统实现方案采用双活接入与异步补偿机制，对外同步接口仅返回必要状态，长耗时任务全部转入任务中心处理，",
        ],
        [
            "测试方案除功能回归外，还需要验证网络闪断、重复消息、数据库切换和缓存失效后的业务连续性表现。",
            "如外部依赖系统字段口径继续变更，将同时影响接口适配、历史数据修复和验收脚本重跑计划，应优先锁定主数据规则。",
            "发布阶段必须先在内部租户完成灰度验证，再逐步放量到目标客户，所有异常请求均需自动打标并进入复盘池。",
        ],
    ),
    "财务文档": _cross_join(
        [
            "本期单据归属{dept}成本中心{cost_center}，预算科目覆盖软件订阅、设备采购、差旅报销和外包服务四类，",
            "申请事项对应{project}项目第{delivery_phase}阶段，涉及{vendor}服务费、测试资源租赁和驻场支持费用，",
            "根据月度滚动预测，当前年度预算总额为{budget_total}万元，已执行{budget_used}万元，剩余可用额度为{budget_balance}万元，",
            "本文件同时关联{invoice_count}张发票与两份补充协议，付款条件、税率和验收节点已由采购、法务和业务部门逐项核对，",
            "为保证费用归集准确，本次申请将按项目维度和职能维度双重入账，涉及跨部门分摊的部分需由财务BP发起二次确认，",
        ],
        [
            "经初审判断，若本次审批通过，不会突破部门年度上限，但要求项目组严格按里程碑提交验收材料和付款依据。",
            "如业务方无法在{acceptance_date}前提供正式验收单，尾款应顺延至下一个付款窗口，并在ERP中挂起处理。",
            "审批通过后，出纳付款、会计入账和预算回写应在同一工作日内完成，确保经营看板与项目台账口径一致。",
        ],
    ),
    "合同协议": _cross_join(
        [
            "双方确认本协议适用于{project}项目相关的技术交付、数据处理和运行保障服务，服务范围以书面确认的需求清单和实施计划为准，",
            "乙方应按照约定提供实施方案、项目人员、运维支持及必要培训材料，任何超出原始范围的新增需求均需经双方签字确认，",
            "合作内容包含系统部署、接口开发、联调支持和验收整改四个阶段，若甲方业务策略发生重大变化，双方应及时重新确认边界，",
            "涉及第三方软件、云资源或硬件设备的部分，由乙方协助甲方完成对接，但第三方服务本身的持续可用性不视为乙方单方保证义务，",
            "双方均应指定唯一接口人负责需求确认、进度同步和问题升级，重大变更指令必须在24小时内形成可归档的书面记录，",
        ],
        [
            "甲方应按里程碑提供业务资料、测试数据和访问权限，乙方则需在关键支持窗口满足约定的响应时限和整改要求。",
            "双方对合作过程中获知的商业计划、客户数据、源代码和审计记录承担保密义务，未经书面许可不得向第三方披露。",
            "若发生严重泄密、恶意毁损数据或连续违约导致合作无法继续，守约方有权解除协议并追究相应经济损失。",
        ],
    ),
    "核心技术": _cross_join(
        [
            "本方案面向{system_name}的关键能力升级，核心目标是在保证现网稳定的前提下完成{algorithm}的在线服务化改造，",
            "整体架构采用离线训练、特征发布、在线推理和审计回放四段式设计，既满足多租户隔离要求，也便于问题复盘，",
            "当前版本重点解决特征时效性、缓存穿透和跨区域部署一致性问题，计划通过统一元数据管理与灰度流量切分提升可控性，",
            "技术路线兼顾性能与可维护性，所有关键模块均要求具备配置热更新、指标埋点和异常熔断能力，",
            "核心推理链路将{feature_count}个特征按静态属性、行为窗口和上下文信号分层编码，在线服务优先命中近端缓存，",
        ],
        [
            "访问控制上采用最小权限模型，研发、运维与安全角色分离管理，涉及模型参数和回放样本的操作均需双人审批。",
            "漏洞治理遵循发现、分级、修复、复测和复盘的闭环流程，高危问题必须在{notice_period}个工作日内完成补丁验证与回归。",
            "部署前需完成接口压测、权限复核、日志抽样和依赖健康检查四项门禁，任一项未通过均不得签署上线确认单。",
        ],
    ),
    "敏感人事": _cross_join(
        [
            "结合{year}年度经营目标与组织盘点结果，当前{dept}条线存在岗位重叠、层级跨度过大和关键岗位继任不足等问题，",
            "近期业务结构调整后，原有编制与薪酬带宽已难以准确反映实际职责差异，如继续沿用旧方案将加大核心岗位流失风险，",
            "董事会在上次专题会上要求人力与业务联合形成可落地的优化方案，重点关注人员结构、成本弹性和关键岗位保留三方面，",
            "截至{review_date}，本轮评估已覆盖{headcount}名正式员工和关键外包岗位，访谈、绩效和市场薪酬样本已完成初步整理，",
            "组织架构调整拟将分散在多个小组的审批、运营和支持职能收拢到共享平台，业务侧仅保留与收入和交付强相关的前线岗位，",
        ],
        [
            "方案执行期间需同步准备沟通口径、权限回收清单和知识交接台账，避免在离岗窗口形成服务中断和合规风险。",
            "如涉及人员优化，补偿原则将按照司龄、绩效和岗位稀缺度综合测算，并提供转岗、推荐和求职支持等配套安排。",
            "落地后一个月内，人力团队需提交复盘报告，评估离职风险、岗位补位效率和成本兑现情况，再决定是否进入第二轮优化。",
        ],
    ),
}

_EXCEL_PROFILES: Dict[str, Dict[str, object]] = {
    "行政通知": {
        "projects": ["{dept}门禁权限梳理", "{dept}培训签到台账", "{dept}办公耗材补充", "{dept}会议室排班优化"],
        "categories": ["行政支出", "培训组织", "办公管理", "场地维护"],
        "statuses": ["已通知", "执行中", "待确认", "已归档"],
        "amount_range": (2000, 80000),
        "remarks": [
            "申请日期：{date}；责任人：{owner}；涉及楼层：{floor}",
            "计划执行日：{date}；接口人：{owner}；成本中心：{cost_center}",
            "整改批次：{doc_no}；确认日期：{review_date}；支持部门：{dept}",
        ],
    },
    "公开报告": {
        "projects": ["{topic}对外材料编制", "{project}案例白皮书发布", "{dept}年度报告校对", "{topic}产品手册修订"],
        "categories": ["品牌传播", "内容审校", "外部发布", "客户说明"],
        "statuses": ["已发布", "待复核", "排版中", "已完成"],
        "amount_range": (5000, 150000),
        "remarks": [
            "披露日期：{date}；审核人：{owner}；版本：v{version}",
            "覆盖客户数：{customer_count}；归档单号：{doc_no}",
            "发布时间窗：{date}；校对责任人：{owner}",
        ],
    },
    "内部备忘": {
        "projects": ["{project}风险跟踪清单", "{project}资源协调事项", "{dept}周报整理", "{topic}会议行动项"],
        "categories": ["内部协同", "会议决议", "资源申请", "风险跟踪"],
        "statuses": ["待处理", "执行中", "已同步", "已关闭"],
        "amount_range": (3000, 120000),
        "remarks": [
            "更新时间：{date}；负责人：{owner}；下次检查：{review_date}",
            "关联会议：{doc_no}；升级状态：{risk_level}",
            "跟踪周期：本周；接口部门：{dept}",
        ],
    },
    "项目文档": {
        "projects": ["{project}需求评审", "{project}集成测试", "{module}灰度发布", "{system_name}缺陷修复"],
        "categories": ["研发投入", "测试执行", "发布保障", "交付准备"],
        "statuses": ["已完成", "执行中", "待验收", "待排期"],
        "amount_range": (10000, 300000),
        "remarks": [
            "里程碑：{milestone_date}；负责人：{owner}；环境：预发",
            "缺陷单：{doc_no}；上线日期：{go_live_date}",
            "验收窗口：{acceptance_date}；模块：{module}",
        ],
    },
    "财务文档": {
        "projects": ["{project}预算申请", "{vendor}采购付款", "{item}费用核销", "{dept}月度成本归集"],
        "categories": ["预算审批", "采购付款", "费用报销", "成本分摊"],
        "statuses": ["已审批", "待审批", "付款中", "已入账"],
        "amount_range": (15000, 800000),
        "remarks": [
            "申请日期：{date}；成本中心：{cost_center}；单据号：{doc_no}",
            "付款节点：{payment_ratio}%预付；供应商：{vendor}",
            "税率：{tax_rate}；验收日期：{acceptance_date}",
        ],
    },
    "合同协议": {
        "projects": ["{vendor}技术合作协议", "{project}保密协议", "{vendor}数据处理协议", "{project}服务等级附录"],
        "categories": ["合同签署", "法务审核", "付款节点", "交付验收"],
        "statuses": ["待法务", "签署中", "已生效", "归档中"],
        "amount_range": (30000, 1200000),
        "remarks": [
            "合同编号：{contract_no}；签署日期：{date}",
            "服务窗口：{service_window}；SLA：{sla_hours}小时",
            "保密期限：{retention_days}日；对接人：{owner}",
        ],
    },
    "核心技术": {
        "projects": ["{algorithm}性能优化", "{system_name}安全加固", "{module}审计改造", "{project}模型回放环境"],
        "categories": ["研发投入", "安全建设", "算力采购", "审计整改"],
        "statuses": ["灰度中", "待验证", "已上线", "待复盘"],
        "amount_range": (50000, 1500000),
        "remarks": [
            "容量目标：{qps}QPS；负责人：{owner}；延迟阈值：{latency_ms}ms",
            "轮换周期：{rotation_days}天；访问组：{access_group}",
            "漏洞编号：{vuln_id}；严重度：{risk_level}",
        ],
    },
    "敏感人事": {
        "projects": ["{dept}岗位调整测算", "{position}薪酬盘点", "{dept}人员安置计划", "{project}继任安排评估"],
        "categories": ["编制管理", "薪酬测算", "补偿预算", "沟通安排"],
        "statuses": ["仅限查阅", "待批准", "执行中", "已归档"],
        "amount_range": (8000, 600000),
        "remarks": [
            "访谈日期：{date}；责任人：{owner}；涉及人数：{headcount_change}",
            "补偿标准：{compensation_months}个月；审批层级：董事会",
            "方案名称：{plan_name}；复核日期：{review_date}",
        ],
    },
}


def _random_date(year: Optional[int] = None) -> str:
    actual_year = year if year is not None else random.choice([2023, 2024, 2025])
    return f"{actual_year}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"


def _fill_template(template: str, replacements: Dict[str, str]) -> str:
    result = template
    for key, value in replacements.items():
        result = result.replace(key, value)
    return result


def _build_base_replacements(category: str) -> Dict[str, str]:
    year = random.choice([2023, 2024, 2025])
    budget_total = random.randint(80, 900)
    budget_used = random.randint(20, budget_total - 10)
    budget_balance = budget_total - budget_used
    topic_pool = _CATEGORY_TOPICS.get(category, ["数据安全", "业务流程优化", "系统升级", "合规审查"])
    return {
        "{month}": str(random.randint(1, 12)),
        "{year}": str(year),
        "{quarter}": random.choice(["Q1", "Q2", "Q3", "Q4"]),
        "{week}": str(random.randint(1, 52)),
        "{version}": f"{random.randint(1, 3)}.{random.randint(0, 9)}",
        "{dept}": random.choice(_DEPTS),
        "{project}": random.choice(_PROJECTS),
        "{sender}": random.choice(_NAMES),
        "{receiver}": random.choice(_NAMES),
        "{author}": random.choice(_NAMES),
        "{approver}": random.choice(_NAMES),
        "{owner}": random.choice(_NAMES),
        "{vendor}": random.choice(_VENDORS),
        "{party}": random.choice(_VENDORS),
        "{party_a}": "本公司",
        "{party_b}": random.choice(_VENDORS),
        "{module}": random.choice(_MODULES),
        "{algorithm}": random.choice(_ALGORITHMS),
        "{system_name}": random.choice(_SYSTEMS),
        "{vuln_id}": f"CVE-2024-{random.randint(10000, 99999)}",
        "{access_group}": random.choice(["核心研发组", "安全委员会", "董事会"]),
        "{date}": _random_date(2024),
        "{review_date}": _random_date(2024),
        "{milestone_date}": _random_date(2024),
        "{go_live_date}": _random_date(2024),
        "{acceptance_date}": _random_date(2024),
        "{doc_no}": f"DOC-{random.randint(10000, 99999)}",
        "{contract_no}": f"HT-{random.randint(2024000, 2024999)}",
        "{item}": random.choice(["差旅费", "设备采购", "培训费用", "咨询服务", "云资源扩容"]),
        "{plan_name}": random.choice(["组织优化方案", "薪酬调整计划", "战略转型方案", "岗位调整方案"]),
        "{next_meeting}": f"{_random_date(2024)} 14:00",
        "{topic}": random.choice(topic_pool),
        "{subject}": random.choice(["项目进展汇报", "风险事项通报", "资源协调", "问题复盘", "排期调整说明"]),
        "{office_location}": random.choice(["上海总部", "北京研发中心", "深圳运营中心", "杭州交付中心"]),
        "{floor}": random.choice(["3层", "5层", "8层", "12层"]),
        "{cost_center}": f"CC-{random.randint(100, 999)}",
        "{delivery_phase}": random.choice(["一期", "二期", "验收前", "扩容期"]),
        "{budget_total}": str(budget_total),
        "{budget_used}": str(budget_used),
        "{budget_balance}": str(budget_balance),
        "{invoice_count}": str(random.randint(4, 24)),
        "{customer_count}": str(random.randint(80, 5000)),
        "{issue_count}": str(random.randint(6, 28)),
        "{headcount}": str(random.randint(12, 180)),
        "{headcount_change}": str(random.randint(5, 35)),
        "{position}": random.choice(["部门总监", "高级研发经理", "财务BP", "运营负责人", "产品负责人"]),
        "{feature_count}": str(random.randint(24, 240)),
        "{payment_ratio}": str(random.choice([10, 20, 30, 40, 50])),
        "{tax_rate}": random.choice(["6%", "9%", "13%"]),
        "{qps}": str(random.randint(3000, 45000)),
        "{latency_ms}": str(random.randint(18, 160)),
        "{rotation_days}": str(random.choice([30, 60, 90])),
        "{retention_days}": str(random.choice([90, 180, 365])),
        "{sla_hours}": str(random.choice([2, 4, 8, 12])),
        "{service_window}": random.choice(["7x24小时", "5x12小时", "工作日8x10小时"]),
        "{risk_level}": random.choice(["高", "中高", "P1", "严重"]),
        "{compensation_months}": str(random.randint(2, 6)),
    }


def _random_fill(template: str, replacements: Optional[Dict[str, str]] = None) -> str:
    """用随机数据填充模板占位符"""
    active_replacements = replacements or _build_base_replacements("")
    return _fill_template(template, active_replacements)


def _build_body_replacements(category: str, base_replacements: Dict[str, str]) -> Dict[str, str]:
    section_count, detail_count = _CATEGORY_SLOT_COUNTS[category]
    section_titles = random.sample(_CATEGORY_SECTION_TITLES[category], section_count)
    paragraphs = random.sample(_CATEGORY_PARAGRAPHS[category], detail_count)
    replacements: Dict[str, str] = {}

    for idx, title in enumerate(section_titles, start=1):
        replacements[f"{{section{idx}}}"] = title
    for idx, paragraph in enumerate(paragraphs, start=1):
        replacements[f"{{detail{idx}}}"] = _fill_template(paragraph, base_replacements)
    return replacements


def _generate_content(level: str) -> Tuple[str, str, str]:
    """生成指定等级的文档标题和内容

    Returns:
        (title, body, category)
    """
    templates = DOCUMENT_TEMPLATES[level]
    tmpl = random.choice(templates)
    category = tmpl["category"]
    base_replacements = _build_base_replacements(category)
    title = _random_fill(random.choice(tmpl["titles"]), base_replacements)
    body_replacements = dict(base_replacements)
    body_replacements.update(_build_body_replacements(category, base_replacements))
    body = _random_fill(tmpl["body_template"], body_replacements)
    return title, body, category


def _create_word_document(title: str, body: str, output_path: str) -> bool:
    """生成 Word 文档（含页眉和样式）"""
    try:
        from docx import Document
        from docx.shared import Pt, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH

        doc = Document()

        section = doc.sections[0]
        header = section.header
        header_para = header.paragraphs[0]
        header_para.text = "内部文档 - 请勿外传"
        header_para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        run = header_para.runs[0]
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor(128, 128, 128)

        doc.add_heading(title, level=1)

        for para_text in body.split("\n"):
            para_text = para_text.strip()
            if not para_text:
                continue
            p = doc.add_paragraph(para_text)
            p.style.font.size = Pt(11)

        doc.save(output_path)
        return True
    except Exception as e:
        print(f"[WARN] Word 生成失败: {e}")
        return False


def _infer_excel_profile(title: str, body: str) -> str:
    text = f"{title}\n{body}"
    keyword_map = [
        ("敏感人事", ["薪酬", "裁员", "组织架构", "董事会", "人事"]),
        ("核心技术", ["算法", "漏洞", "密钥", "安全", "推理", "审计"]),
        ("合同协议", ["协议", "合同", "甲方", "乙方", "保密"]),
        ("财务文档", ["财务", "预算", "报销", "采购", "付款", "审批"]),
        ("项目文档", ["需求规格说明书", "测试报告", "架构设计", "建设目标", "发布"]),
        ("内部备忘", ["备忘录", "会议纪要", "内部沟通", "行动安排", "下次会议"]),
        ("公开报告", ["社会责任报告", "白皮书", "产品使用手册", "治理进展", "公开"]),
        ("行政通知", ["通知", "考勤", "办公区域", "员工手册", "假期安排"]),
    ]
    for profile, keywords in keyword_map:
        if any(keyword in text for keyword in keywords):
            return profile
    return "项目文档"


def _build_excel_rows(profile: str) -> List[Tuple[str, str, float, str, str]]:
    cfg = _EXCEL_PROFILES.get(profile, _EXCEL_PROFILES["项目文档"])
    rows: List[Tuple[str, str, float, str, str]] = []
    row_count = random.randint(12, 24)

    for _ in range(row_count):
        replacements = _build_base_replacements(profile)
        project_name = _fill_template(random.choice(cfg["projects"]), replacements)
        category = random.choice(cfg["categories"])
        amount_min, amount_max = cfg["amount_range"]
        amount = round(random.uniform(amount_min, amount_max), 2)
        status = random.choice(cfg["statuses"])
        remark = _fill_template(random.choice(cfg["remarks"]), replacements)
        rows.append((project_name, category, amount, status, remark))

    return rows


def _create_excel_document(title: str, body: str, output_path: str) -> bool:
    """生成 Excel 文档（含格式化）"""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

        wb = Workbook()
        ws = wb.active
        ws.title = "数据"
        ws.freeze_panes = "A4"

        thin_border = Border(
            left=Side(style="thin", color="D9D9D9"),
            right=Side(style="thin", color="D9D9D9"),
            top=Side(style="thin", color="D9D9D9"),
            bottom=Side(style="thin", color="D9D9D9"),
        )

        ws.merge_cells("A1:F1")
        cell = ws["A1"]
        cell.value = title
        cell.font = Font(size=14, bold=True)
        cell.alignment = Alignment(horizontal="center")

        headers = ["序号", "项目", "类别", "金额", "状态", "备注"]
        header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=3, column=col, value=header)
            cell.font = Font(color="FFFFFF", bold=True)
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center")
            cell.border = thin_border

        profile = _infer_excel_profile(title, body)
        for idx, row_data in enumerate(_build_excel_rows(profile), start=1):
            row = idx + 3
            project_name, category, amount, status, remark = row_data
            ws.cell(row=row, column=1, value=idx).border = thin_border
            ws.cell(row=row, column=2, value=project_name).border = thin_border
            ws.cell(row=row, column=3, value=category).border = thin_border
            amount_cell = ws.cell(row=row, column=4, value=amount)
            amount_cell.number_format = "#,##0.00"
            amount_cell.border = thin_border
            ws.cell(row=row, column=5, value=status).border = thin_border
            ws.cell(row=row, column=6, value=remark).border = thin_border

        ws.column_dimensions["A"].width = 8
        ws.column_dimensions["B"].width = 24
        ws.column_dimensions["C"].width = 16
        ws.column_dimensions["D"].width = 14
        ws.column_dimensions["E"].width = 12
        ws.column_dimensions["F"].width = 36

        wb.save(output_path)
        return True
    except Exception as e:
        print(f"[WARN] Excel 生成失败: {e}")
        return False


def _resolve_pdf_font() -> str:
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        candidates = [
            ("SimHei", "C:/Windows/Fonts/simhei.ttf"),
            ("MicrosoftYaHei", "C:/Windows/Fonts/msyh.ttc"),
            ("SimSun", "C:/Windows/Fonts/simsun.ttc"),
        ]
        for font_name, font_path in candidates:
            if not os.path.exists(font_path):
                continue
            try:
                pdfmetrics.getFont(font_name)
            except KeyError:
                pdfmetrics.registerFont(TTFont(font_name, font_path))
            return font_name
    except Exception:
        pass
    return "Helvetica"


def _create_pdf_document(title: str, body: str, output_path: str) -> bool:
    """生成原生 PDF 文档"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

        font_name = _resolve_pdf_font()
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
            leftMargin=2.5 * cm,
            rightMargin=2.5 * cm,
        )
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Title"],
            fontName=font_name,
            fontSize=16,
            leading=20,
            spaceAfter=20,
        )
        body_style = ParagraphStyle(
            "DocBody",
            parent=styles["Normal"],
            fontName=font_name,
            fontSize=11,
            leading=16,
        )

        story = []
        safe_title = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(safe_title, title_style))
        story.append(Spacer(1, 0.5 * cm))

        for para in body.split("\n"):
            para = para.strip()
            if not para:
                continue
            safe = para.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(safe, body_style))
            story.append(Spacer(1, 0.2 * cm))

        doc.build(story)
        return True
    except Exception as e:
        print(f"[WARN] PDF 生成失败: {e}")
        return False


def _add_watermark(input_pdf: str, output_pdf: str, opacity: float = 0.3) -> bool:
    """为 PDF 添加文字水印"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas as reportlab_canvas
        import io

        watermark_buf = io.BytesIO()
        c = reportlab_canvas.Canvas(watermark_buf, pagesize=A4)
        c.saveState()
        c.setFillAlpha(opacity)
        c.setFillColorRGB(0.8, 0.8, 0.8)
        c.setFont("Helvetica", 50)
        c.translate(A4[0] / 2, A4[1] / 2)
        c.rotate(45)
        c.drawCentredString(0, 0, "CONFIDENTIAL")
        c.restoreState()
        c.save()
        watermark_buf.seek(0)

        from PyPDF2 import PdfReader, PdfWriter

        watermark_reader = PdfReader(watermark_buf)
        watermark_page = watermark_reader.pages[0]

        reader = PdfReader(input_pdf)
        writer = PdfWriter()

        for page in reader.pages:
            page.merge_page(watermark_page)
            writer.add_page(page)

        with open(output_pdf, "wb") as f:
            writer.write(f)
        return True
    except Exception as e:
        print(f"[WARN] 水印添加失败: {e}")
        return False


def generate_custom_dataset(
    output_dir: str = "data/custom",
    seed: int = 42,
) -> str:
    """生成 1300 份自建数据集

    Args:
        output_dir: 输出目录
        seed: 随机种子

    Returns:
        生成的 labels.csv 路径
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    generation_plan: List[Tuple[str, str, int]] = [
        ("docx", "L1", 80), ("docx", "L2", 150),
        ("docx", "L3", 150), ("docx", "L4", 120),
        ("xlsx", "L1", 60), ("xlsx", "L2", 90),
        ("xlsx", "L3", 90), ("xlsx", "L4", 60),
        ("pdf", "L1", 80), ("pdf", "L2", 80),
        ("pdf", "L3", 80), ("pdf", "L4", 60),
        ("watermark_pdf", "L1", 60), ("watermark_pdf", "L2", 60),
        ("watermark_pdf", "L3", 50), ("watermark_pdf", "L4", 30),
    ]

    csv_path = os.path.join(output_dir, "labels.csv")
    records: List[Dict] = []
    doc_idx = 0

    for fmt, level, count in generation_plan:
        for _ in range(count):
            doc_id = f"custom_{doc_idx:05d}"
            title, body, category = _generate_content(level)

            if fmt == "docx":
                fname = f"{doc_id}.docx"
                fpath = os.path.join(output_dir, fname)
                ok = _create_word_document(title, body, fpath)
            elif fmt == "xlsx":
                fname = f"{doc_id}.xlsx"
                fpath = os.path.join(output_dir, fname)
                ok = _create_excel_document(title, body, fpath)
            elif fmt == "pdf":
                fname = f"{doc_id}.pdf"
                fpath = os.path.join(output_dir, fname)
                ok = _create_pdf_document(title, body, fpath)
            elif fmt == "watermark_pdf":
                tmp_path = os.path.join(output_dir, f"{doc_id}_tmp.pdf")
                fname = f"{doc_id}_wm.pdf"
                fpath = os.path.join(output_dir, fname)
                ok = _create_pdf_document(title, body, tmp_path)
                if ok:
                    opacity = round(random.uniform(0.2, 0.5), 2)
                    ok = _add_watermark(tmp_path, fpath, opacity)
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            else:
                continue

            records.append({
                "doc_id": doc_id,
                "filename": fname,
                "format": fmt if fmt != "watermark_pdf" else "watermark_pdf",
                "category": category,
                "sensitivity_level": level,
                "title": title,
                "generated": int(ok),
            })
            doc_idx += 1

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "doc_id", "filename", "format", "category",
            "sensitivity_level", "title", "generated",
        ])
        writer.writeheader()
        writer.writerows(records)

    level_counts = {}
    format_counts = {}
    success = 0
    for record in records:
        level_counts[record["sensitivity_level"]] = level_counts.get(record["sensitivity_level"], 0) + 1
        format_counts[record["format"]] = format_counts.get(record["format"], 0) + 1
        success += record["generated"]

    print("[自建数据集] 生成完成:")
    print(f"  总计: {len(records)} 份 (成功: {success})")
    print(f"  等级分布: {level_counts}")
    print(f"  格式分布: {format_counts}")
    print(f"  标签文件: {csv_path}")
    return csv_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="自建多格式文档数据集生成")
    parser.add_argument(
        "--output-dir", default="data/custom", help="输出目录",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_custom_dataset(output_dir=args.output_dir, seed=args.seed)
