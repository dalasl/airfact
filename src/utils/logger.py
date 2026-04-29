"""日志工具"""

import os
import sys
from datetime import datetime


def setup_logger(name: str = "dlp-profiling", log_dir: str = "logs/"):
    """配置日志"""
    try:
        from loguru import logger

        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

        logger.remove()
        logger.add(sys.stderr, level="INFO",
                    format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | {message}")
        logger.add(log_file, level="DEBUG", rotation="50 MB")

        return logger
    except ImportError:
        import logging
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
        logger.addHandler(handler)
        return logger
