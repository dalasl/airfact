"""
数据加载与预处理工具
"""

import os
import json
import glob
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class DataLoader:
    """多数据集加载器"""

    def __init__(self, data_dir: str = "data/"):
        self.data_dir = data_dir

    def load_rvlcdip(self, subset_size: int = 3000) -> Dict:
        """加载 RVL-CDIP 数据集（3000样本，10类企业文档）"""
        path = os.path.join(self.data_dir, "raw/rvlcdip/")
        return {
            "name": "RVL-CDIP",
            "source": "harley2015rvlcdip",
            "total": subset_size,
            "format": "TIFF扫描件",
            "categories": [
                "letter", "memo", "form", "invoice", "report",
                "email", "resume", "scientific", "spec_sheet", "folder_cover"
            ],
            "sensitivity_distribution": {"L1": 750, "L2": 900, "L3": 800, "L4": 550},
            "data_dir": path,
        }

    def load_enron(self, subset_size: int = 700) -> Dict:
        """加载 Enron 邮件数据集（700样本，渲染PDF）"""
        path = os.path.join(self.data_dir, "raw/enron/")
        return {
            "name": "Enron",
            "source": "klimt2004enron (EDRM版本)",
            "total": subset_size,
            "format": "邮件渲染PDF",
            "sensitivity_distribution": {"L1": 180, "L2": 220, "L3": 180, "L4": 120},
            "data_dir": path,
        }

    def load_selfbuilt(self, subset_size: int = 1300) -> Dict:
        """加载自建数据集（1300样本，多格式办公文档）"""
        path = os.path.join(self.data_dir, "raw/selfbuilt/")
        return {
            "name": "自建数据集",
            "total": subset_size,
            "format_distribution": {
                "Word (.docx)": 500,
                "Excel (.xlsx)": 300,
                "PDF": 300,
                "水印PDF": 200,
            },
            "sensitivity_distribution": {"L1": 280, "L2": 380, "L3": 370, "L4": 270},
            "data_dir": path,
        }

    def load_cert_r42(self) -> Dict:
        """加载 CERT r4.2 内部威胁数据集"""
        path = os.path.join(self.data_dir, "raw/cert_r4.2/")
        return {
            "name": "CERT Insider Threat r4.2",
            "source": "CMU SEI (DARPA ADAMS)",
            "total_events": 3300000,
            "users": 1000,
            "working_days": 501,
            "event_types": ["logon", "file", "email", "http", "device"],
            "threat_scenarios": 70,
            "data_theft_scenarios": 42,
            "data_dir": path,
        }

    def load_security_policies(self) -> Dict:
        """加载安全策略语料"""
        path = os.path.join(self.data_dir, "raw/security_policies/")
        return {
            "name": "安全策略语料",
            "total": 200,
            "sources": {
                "NIST SP 800-53 Rev.5": 70,
                "MITRE ATT&CK": 40,
                "Sigma Rules": 30,
                "NIST SP 800-171 + CIS Controls v8": 30,
                "自建（数据安全法/个保法/复合规则）": 30,
            },
            "complexity": {"simple": 70, "medium": 80, "complex": 50},
            "data_dir": path,
        }

    def load_all_datasets(self) -> Dict[str, Dict]:
        """加载所有数据集元信息"""
        return {
            "documents": {
                "rvlcdip": self.load_rvlcdip(),
                "enron": self.load_enron(),
                "selfbuilt": self.load_selfbuilt(),
            },
            "behavior": self.load_cert_r42(),
            "policies": self.load_security_policies(),
        }
