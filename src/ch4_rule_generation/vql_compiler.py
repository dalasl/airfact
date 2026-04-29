"""
VQL 编译器

将四元组 IR 编译为可在 Velociraptor 上执行的 VQL 脚本。
支持语法检查、逻辑冲突检测和执行测试。

VQL (Velociraptor Query Language) 特点：
- SQL-like 声明式语法
- 插件机制提供文件系统、进程、网络等访问能力
- 脚本在终端本地执行，敏感数据不上传（数据最小化原则）

编译路径（按优先级）：
1. Jinja2 模板渲染（模板合成引擎提供 VQL 模板时）
2. IR 直接编译（无匹配模板时的回退路径）
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from jinja2 import BaseLoader, Environment, TemplateSyntaxError, Undefined

from .quadruple_ir import QuadrupleIR, ConstraintOperator


class _SilentUndefined(Undefined):
    """未定义变量渲染为空字符串，防止模板残留占位符"""

    def __str__(self):
        return ""

    def __iter__(self):
        return iter([])

    def __bool__(self):
        return False


@dataclass
class VQLScript:
    """VQL 脚本"""
    script: str
    template_id: str = ""
    policy_text: str = ""
    syntax_valid: bool = False
    logic_valid: bool = False
    estimated_memory_kb: int = 0
    estimated_instructions: int = 0


class VQLCompiler:
    """VQL 编译器

    将四元组中间表示编译为 VQL 脚本。
    支持 Jinja2 模板渲染和 IR 直接编译两条路径。
    """

    # VQL 插件映射 — 使用 Velociraptor 真实插件
    # watch_usn: NTFS USN Journal 监控（返回 TimeStamp, Filename, FullPath, Reason）
    # watch_etw: ETW 事件跟踪（返回 System.TimeStamp, System.ProcessID, EventData）
    # diff + glob: 周期性文件系统差异比对
    PLUGIN_MAP = {
        "read": {
            "source": "watch_usn(device='C:')",
            "reason_filter": "DATA_EXTEND|DATA_OVERWRITE",
            "columns": ["TimeStamp", "Filename", "FullPath", "Reason"],
        },
        "write": {
            "source": "watch_usn(device='C:')",
            "reason_filter": "DATA_EXTEND|DATA_OVERWRITE|DATA_TRUNCATION",
            "columns": ["TimeStamp", "Filename", "FullPath", "Reason"],
        },
        "copy": {
            "source": "watch_usn(device='C:')",
            "reason_filter": "FILE_CREATE|CLOSE",
            "columns": ["TimeStamp", "Filename", "FullPath", "Reason"],
        },
        "upload": {
            "source": "watch_etw(guid='{7DD42A49-5329-4832-8DFD-43D979153A88}', any_keyword=0x80)",
            "reason_filter": None,
            "columns": ["System.TimeStamp AS TimeStamp",
                        "System.ProcessID AS Pid",
                        "EventData.daddr AS DestAddr",
                        "EventData.dport AS DestPort"],
        },
        "send": {
            "source": "watch_etw(guid='{49C2C27C-FE2D-40BF-8C4E-C3FB518037E7}')",
            "reason_filter": None,
            "columns": ["System.TimeStamp AS TimeStamp",
                        "System.ProcessID AS Pid",
                        "EventData"],
        },
        "print": {
            "source": "watch_etw(guid='{747EF6FD-E535-4D16-B510-42C90F6873A1}')",
            "reason_filter": None,
            "columns": ["System.TimeStamp AS TimeStamp",
                        "System.ProcessID AS Pid",
                        "EventData.param1 AS DocumentName"],
        },
        "download": {
            "source": "watch_usn(device='C:')",
            "reason_filter": "FILE_CREATE|DATA_EXTEND",
            "columns": ["TimeStamp", "Filename", "FullPath", "Reason"],
        },
        "usb": {
            "source": ("diff(query={"
                       "SELECT FullPath, Size, Mtime "
                       "FROM glob(globs='D:\\\\**')"
                       "}, key='FullPath', period=10)"),
            "reason_filter": None,
            "columns": ["FullPath", "Size", "Mtime", "Diff"],
        },
    }

    # 操作符映射
    OP_MAP = {
        ConstraintOperator.EQ: "=",
        ConstraintOperator.NEQ: "!=",
        ConstraintOperator.GT: ">",
        ConstraintOperator.GTE: ">=",
        ConstraintOperator.LT: "<",
        ConstraintOperator.LTE: "<=",
        ConstraintOperator.IN: "IN",
        ConstraintOperator.NOT_IN: "NOT IN",
        ConstraintOperator.CONTAINS: "=~",
        ConstraintOperator.MATCHES: "=~",
    }

    # 敏感等级数值映射
    LEVEL_MAP = {"L1": 1, "L2": 2, "L3": 3, "L4": 4}

    def __init__(self):
        self._jinja_env = Environment(
            loader=BaseLoader(),
            autoescape=False,
            keep_trailing_newline=True,
            undefined=_SilentUndefined,
        )

    def compile(self, ir: QuadrupleIR, template_vql: str = None) -> VQLScript:
        """编译四元组为 VQL 脚本

        Args:
            ir: 四元组中间表示
            template_vql: 预匹配的模板 VQL（来自模板合成引擎）

        Returns:
            VQL 脚本对象
        """
        if template_vql:
            vql = self.render_template(template_vql, ir)
            script = VQLScript(
                script=vql,
                policy_text=ir.policy_text,
            )
        else:
            vql = self._compile_from_ir(ir)
            script = VQLScript(
                script=vql,
                policy_text=ir.policy_text,
            )

        script.syntax_valid = self.check_syntax(script.script)
        script.logic_valid = not self.detect_logic_conflicts(script.script)

        return script

    def render_template(self, template_vql: str, ir: QuadrupleIR) -> str:
        """Jinja2 模板渲染

        将模板中的 {placeholder} 和 {{ jinja_var }} 占位符
        替换为 IR 约束值。

        支持两种占位符风格：
        1. 简单替换：{sensitivity} → Jinja2 {{ sensitivity }}
        2. 原生 Jinja2：{{ var }}, {% if %} 等

        Args:
            template_vql: VQL 模板字符串（含占位符）
            ir: 四元组中间表示

        Returns:
            渲染后的 VQL 脚本
        """
        context = self._build_template_context(ir)

        # 将 {placeholder} 风格转为 Jinja2 {{ placeholder }}
        normalized = re.sub(
            r"(?<!\{)\{(\w+)\}(?!\})",
            r"{{ \1 }}",
            template_vql,
        )

        try:
            tmpl = self._jinja_env.from_string(normalized)
            rendered = tmpl.render(**context)
        except TemplateSyntaxError:
            rendered = self._fallback_render(template_vql, context)

        # 清理残留空占位符
        rendered = re.sub(r"\s*(?:AND|OR)\s+(?:''|\"\")\s*", "", rendered)
        rendered = re.sub(r"\s{2,}", " ", rendered).strip()

        return rendered

    def _build_template_context(self, ir: QuadrupleIR) -> Dict[str, str]:
        """从 IR 约束构建模板渲染上下文"""
        ctx: Dict[str, str] = {}

        for c in ir.subject.constraints:
            ctx[f"subject_{c.attribute}"] = str(c.value)
            if c.attribute == "role":
                ctx["authorized_role"] = str(c.value)
                ctx["authorized_roles"] = f"'{c.value}'"

        for c in ir.obj.constraints:
            ctx[f"object_{c.attribute}"] = str(c.value)
            if c.attribute == "sensitivity":
                ctx["sensitivity"] = str(c.value)
                ctx["min_sensitivity"] = str(c.value)
            elif c.attribute == "type":
                ctx["file_type"] = str(c.value)

        for c in ir.action.constraints:
            ctx["action"] = c.attribute

        for c in ir.context.constraints:
            ctx[f"context_{c.attribute}"] = str(c.value)
            if c.attribute == "network":
                ctx["network_env"] = str(c.value)
            elif c.attribute == "device":
                ctx["device_type"] = str(c.value)

        # 常用默认值
        ctx.setdefault("threshold", "10")
        ctx.setdefault("time_window", "1h")
        ctx.setdefault("max_depth", "3")
        ctx.setdefault("min_channels", "2")
        ctx.setdefault("max_time_window", "1h")
        ctx.setdefault("internal_domains", "'company.com'")
        ctx.setdefault("whitelisted_urls", "''")
        ctx.setdefault("time_constraint", "")

        return ctx

    @staticmethod
    def _fallback_render(template_vql: str, context: Dict[str, str]) -> str:
        """Jinja2 解析失败时的简单字符串替换回退"""
        result = template_vql
        for key, value in context.items():
            result = result.replace(f"{{{key}}}", value)
        # 清理未替换的占位符
        result = re.sub(r"\{[^}]+\}", "''", result)
        return result

    # 文件类型到扩展名正则的映射
    _FILE_TYPE_REGEX = {
        "PDF": r"\.pdf$",
        "DOCX": r"\.docx?$",
        "XLSX": r"\.xlsx?$",
        "PPTX": r"\.pptx?$",
        "CSV": r"\.csv$",
        "TXT": r"\.txt$",
    }

    def _compile_from_ir(self, ir: QuadrupleIR) -> str:
        """从 IR 直接编译 VQL（无模板匹配时的回退路径）

        生成使用 Velociraptor 真实插件的 VQL 脚本：
        - watch_usn: 文件操作监控（NTFS USN Journal）
        - watch_etw: 网络/邮件/打印事件跟踪（ETW）
        - diff + glob: USB 等外设文件变化检测
        """
        # 确定动作类型和对应的插件配置
        action_key = None
        for c in ir.action.constraints:
            if c.attribute in self.PLUGIN_MAP:
                action_key = c.attribute
                break

        # USB 设备上下文覆盖
        for c in ir.context.constraints:
            if c.attribute == "device" and c.value == "usb":
                action_key = "usb"

        # 下载动作的路径特化
        is_download = action_key == "download"

        # 获取插件配置，默认使用 watch_usn 文件监控
        plugin_cfg = self.PLUGIN_MAP.get(action_key, self.PLUGIN_MAP["read"])
        source = plugin_cfg["source"]
        columns = plugin_cfg["columns"]
        reason_filter = plugin_cfg.get("reason_filter")

        # 构建 WHERE 条件
        conditions = []

        # 基于真实插件列的 Reason 过滤
        if reason_filter:
            conditions.append(f"Reason =~ '{reason_filter}'")

        # 下载操作限定 Downloads 目录
        if is_download:
            conditions.append(r"FullPath =~ 'Downloads'")

        # USB diff 过滤新增文件
        if action_key == "usb":
            conditions.append("Diff = 'added'")

        # 客体约束 → 文件路径/扩展名过滤
        for c in ir.obj.constraints:
            if c.attribute == "type":
                ext_regex = self._FILE_TYPE_REGEX.get(str(c.value).upper())
                if ext_regex:
                    conditions.append(f"FullPath =~ '{ext_regex}'")
                else:
                    conditions.append(f"FullPath =~ '\\.{c.value}$'")
            elif c.attribute == "path":
                conditions.append(f"FullPath =~ '{c.value}'")

        # 上下文约束 → 时间条件
        for c in ir.context.constraints:
            if c.attribute == "work_hours":
                if not c.value:
                    conditions.append(
                        "NOT (timestamp(epoch=TimeStamp).Hour >= 9 "
                        "AND timestamp(epoch=TimeStamp).Hour <= 18)"
                    )
                else:
                    conditions.append(
                        "timestamp(epoch=TimeStamp).Hour >= 9 "
                        "AND timestamp(epoch=TimeStamp).Hour <= 18"
                    )

        where_clause = "\n    AND ".join(conditions) if conditions else "TRUE"

        # 动作名称（用于注释标识）
        action_name = "GenericMonitor"
        for c in ir.action.constraints:
            action_name = c.attribute.title()
            break

        lines = [
            f"// DLP.AutoGenerated.{action_name}",
            f"LET events = SELECT {', '.join(columns)}",
            f"  FROM {source}",
            f"  WHERE {where_clause}",
            "SELECT * FROM events",
        ]

        return "\n".join(lines)

    @staticmethod
    def check_syntax(vql: str) -> bool:
        """VQL 语法检查

        验证 Velociraptor VQL 结构：
        - SELECT ... FROM ... [WHERE ...]
        - LET ... = SELECT ... FROM ...
        - 括号/引号平衡
        """
        # 去除注释行
        lines = [l for l in vql.split("\n") if not l.strip().startswith("//")]
        vql_clean = " ".join(lines)
        vql_upper = vql_clean.upper().strip()

        # 必须包含 SELECT 和 FROM
        if "SELECT" not in vql_upper:
            return False
        if "FROM" not in vql_upper:
            return False

        # 括号平衡
        if vql_clean.count("(") != vql_clean.count(")"):
            return False

        # 单引号平衡
        if vql_clean.count("'") % 2 != 0:
            return False

        return True

    @staticmethod
    def detect_logic_conflicts(vql: str) -> bool:
        """逻辑冲突检测

        检测互斥条件（如 X > 5 AND X < 3）。

        Returns:
            True 表示存在冲突
        """
        # 简化版冲突检测
        patterns = [
            # 同一字段的矛盾比较
            r"(\w+)\s*>\s*(\d+).*\1\s*<\s*(\d+)",
            r"(\w+)\s*=\s*'([^']+)'.*\1\s*!=\s*'\2'",
        ]

        for pattern in patterns:
            match = re.search(pattern, vql)
            if match:
                return True

        return False

    def batch_compile(self, ir_list: List[QuadrupleIR]) -> List[VQLScript]:
        """批量编译"""
        return [self.compile(ir) for ir in ir_list]
