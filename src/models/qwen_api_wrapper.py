"""
Qwen API 推理封装

通过 DashScope / OpenAI 兼容 API 调用大模型，无需本地 GPU。
与 QwenWrapper 保持相同的 (prompt: str, seed: int) -> str 接口，
可直接替换注入到 GradingPipeline.llm_fn。

MC-Dropout 模拟：通过 temperature + seed 控制采样随机性，
不同 seed 产生不同输出，实现自洽性校验所需的多次采样。

用法:
    # DashScope（阿里云百炼）
    wrapper = QwenAPIWrapper(
        api_key="sk-xxx",
        model="qwen-plus",
        backend="dashscope",
    )

    # OpenAI 兼容 API（vLLM / Ollama / LiteLLM 等）
    wrapper = QwenAPIWrapper(
        api_key="sk-xxx",
        model="qwen2.5-7b-instruct",
        backend="openai",
        base_url="http://localhost:8000/v1",
    )

    result = wrapper("请对以下文件进行敏感分级...", seed=42)
"""

import hashlib
import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = "你是企业数据安全专家。"


class QwenAPIWrapper:
    """Qwen API 推理引擎

    Args:
        api_key: API 密钥（也可通过环境变量 DASHSCOPE_API_KEY 或 OPENAI_API_KEY 设置）
        model: 模型名称
        backend: API 后端类型 ("dashscope" 或 "openai")
        base_url: OpenAI 兼容 API 的地址（仅 backend="openai" 时需要）
        max_tokens: 最大生成 token 数
        temperature: 采样温度
        top_p: 核采样概率
        system_prompt: 系统提示词
        max_retries: 最大重试次数
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "qwen-plus",
        backend: str = "dashscope",
        base_url: str = "",
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system_prompt: str = _SYSTEM_PROMPT,
        max_retries: int = 3,
    ):
        self.model = model
        self.backend = backend
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt
        self.max_retries = max_retries

        # API key resolution
        if api_key:
            self.api_key = api_key
        elif backend == "dashscope":
            self.api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        else:
            self.api_key = os.environ.get("OPENAI_API_KEY", "")

    def generate(self, prompt: str, seed: int = 0) -> str:
        """调用 API 生成文本

        Args:
            prompt: 输入提示文本
            seed: 随机种子（通过 seed 参数传递给 API，模拟 MC-Dropout 多样性）

        Returns:
            模型生成的文本
        """
        if self.backend == "dashscope":
            return self._call_dashscope(prompt, seed)
        else:
            return self._call_openai_compatible(prompt, seed)

    def _call_dashscope(self, prompt: str, seed: int) -> str:
        """DashScope API 调用（阿里云百炼平台）"""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai 库未安装。安装: pip install openai"
            )

        client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    seed=seed,
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"[QwenAPI] DashScope attempt {attempt+1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

        return self._fallback_response(prompt, seed)

    def _call_openai_compatible(self, prompt: str, seed: int) -> str:
        """OpenAI 兼容 API 调用（vLLM / Ollama / LiteLLM 等）"""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai 库未安装。安装: pip install openai"
            )

        client = OpenAI(
            api_key=self.api_key or "not-needed",
            base_url=self.base_url or "http://localhost:8000/v1",
        )

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    seed=seed,
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"[QwenAPI] OpenAI-compat attempt {attempt+1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

        return self._fallback_response(prompt, seed)

    @staticmethod
    def _fallback_response(prompt: str, seed: int) -> str:
        """API 不可达时的降级响应"""
        logger.error("[QwenAPI] All retries exhausted, returning fallback")
        h = int(hashlib.md5(f"{prompt}{seed}".encode()).hexdigest()[:8], 16)
        levels = ["L1", "L2", "L3", "L4"]
        level = levels[h % 4]
        return json.dumps({
            "level": level,
            "category": "fallback",
            "reason": "API 调用失败，降级为哈希分级",
            "confidence": 0.5,
        })

    def __call__(self, prompt: str, seed: int = 0) -> str:
        """兼容 GradingPipeline.llm_fn 接口"""
        return self.generate(prompt, seed)
