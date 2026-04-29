"""
Qwen2.5-7B 推理封装

提供 4-bit AWQ 量化加载、MC-Dropout 采样、种子控制的统一推理接口。
底层依赖: transformers, autoawq/auto-gptq

论文对应：
    - 第3章认知层 LLM 推理
    - MC-Dropout 不确定性量化（算法 alg:context-grading 阶段1）
"""

import torch
from typing import Optional


_DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


class QwenWrapper:
    """Qwen2.5-7B 推理引擎

    支持两种推理模式：
    1. 标准推理：dropout 关闭，确定性输出
    2. MC-Dropout 推理：dropout 保持激活，每次 seed 产生不同掩码

    Args:
        model_name: HuggingFace 模型名称或本地路径
        quantization: 量化方式 ("awq", "gptq", "none")
        max_new_tokens: 最大生成 token 数
        device: 计算设备
        mc_dropout: 是否启用 MC-Dropout 模式
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        quantization: str = "awq",
        max_new_tokens: int = 512,
        device: str = "cuda",
        mc_dropout: bool = True,
    ):
        self.model_name = model_name
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.mc_dropout = mc_dropout
        self._model = None
        self._tokenizer = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self):
        """加载模型到指定设备"""
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True,
        )

        load_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto" if self.device == "cuda" else None,
        }

        if self.quantization == "awq":
            load_kwargs["torch_dtype"] = torch.float16
        elif self.quantization == "gptq":
            load_kwargs["torch_dtype"] = torch.float16
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **load_kwargs,
        )

        if self.device != "cuda" and load_kwargs.get("device_map") is None:
            self._model = self._model.to(self.device)

    def unload(self):
        """释放模型显存"""
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _enable_mc_dropout(self):
        """激活模型中所有 Dropout 层（MC-Dropout 核心）

        4-bit 量化权重保持冻结，仅 Dropout 掩码在不同 seed 间变化。
        """
        for module in self._model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()

    def _disable_mc_dropout(self):
        """关闭所有 Dropout 层"""
        self._model.eval()

    def generate(self, prompt: str, seed: int = 0) -> str:
        """单次推理

        Args:
            prompt: 输入提示文本
            seed: 随机种子（MC-Dropout 模式下控制 Dropout 掩码）

        Returns:
            模型生成的文本
        """
        self.load()

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        if self.mc_dropout:
            self._enable_mc_dropout()
        else:
            self._disable_mc_dropout()

        messages = [
            {"role": "system", "content": "你是企业数据安全专家。"},
            {"role": "user", "content": prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = outputs[0][input_len:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    def __call__(self, prompt: str, seed: int = 0) -> str:
        """兼容 GradingPipeline.llm_fn 接口"""
        return self.generate(prompt, seed)
