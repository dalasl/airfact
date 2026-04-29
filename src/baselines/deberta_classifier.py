"""
DeBERTa-v3-base 文档敏感等级分类基线

论文 5.4.1 表 tab:grading-compare 对比基线之一。
使用 DeBERTa-v3-base 在文档数据集上微调 L1-L4 四分类。

此基线代表纯监督学习方法：
- 优点：端到端训练，无需特征工程
- 缺点：需要标注数据，跨域泛化差，无法解释

对应论文表 5-4 "DeBERTa-ft" 行。

训练超参数（论文 5.1.2）：
    lr=2e-5, batch=16, epoch=5, AdamW, 10% linear warmup
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


LEVEL_TO_ID = {"L1": 0, "L2": 1, "L3": 2, "L4": 3}
ID_TO_LEVEL = {v: k for k, v in LEVEL_TO_ID.items()}


@dataclass
class DeBERTaGradingResult:
    """DeBERTa 分级结果

    与 GradingDecision 对齐的接口。
    """
    level: str
    confidence: float
    consistency_ratio: float = 1.0
    uncertainty: float = 0.0
    is_accepted: bool = True
    needs_human_review: bool = False
    reason: str = ""
    all_predictions: List[str] = None
    logits: Optional[np.ndarray] = None


class DocumentDataset:
    """文档分类数据集"""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class DeBERTaClassifier:
    """DeBERTa-v3-base 文档分类器

    Args:
        model_name: HuggingFace 模型名称
        num_labels: 分类数（4: L1-L4）
        max_length: 最大序列长度
        device: 推理设备
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        num_labels: int = 4,
        max_length: int = 512,
        device: str = None,
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length

        self._model = None
        self._tokenizer = None

        if device:
            self._device = device
        elif _HAS_TORCH and torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

    def _load_model(self):
        """延迟加载模型"""
        if self._model is not None:
            return

        if not _HAS_TRANSFORMERS:
            raise ImportError(
                "transformers is required for DeBERTa baseline. "
                "Install with: pip install transformers"
            )

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
        ).to(self._device)

    def load_checkpoint(self, checkpoint_path: str):
        """加载微调后的检查点"""
        if not _HAS_TORCH:
            return

        self._load_model()
        state_dict = torch.load(checkpoint_path, map_location=self._device)
        self._model.load_state_dict(state_dict)
        self._model.eval()

    def train(
        self,
        train_texts: List[str],
        train_labels: List[str],
        val_texts: List[str] = None,
        val_labels: List[str] = None,
        epochs: int = 5,
        batch_size: int = 16,
        lr: float = 2e-5,
        warmup_ratio: float = 0.1,
        output_dir: str = "checkpoints/deberta",
    ) -> Dict[str, List[float]]:
        """微调训练

        Args:
            train_texts: 训练文本
            train_labels: 训练标签（"L1"-"L4"）
            val_texts: 验证文本
            val_labels: 验证标签
            epochs: 训练轮次
            batch_size: 批大小
            lr: 学习率
            warmup_ratio: 预热比例
            output_dir: 检查点保存目录

        Returns:
            训练历史 {train_loss, val_loss, val_accuracy}
        """
        if not _HAS_TORCH or not _HAS_TRANSFORMERS:
            raise ImportError("torch and transformers required for training")

        self._load_model()
        self._model.train()

        train_label_ids = [LEVEL_TO_ID[l] for l in train_labels]
        train_dataset = DocumentDataset(
            train_texts, train_label_ids, self._tokenizer, self.max_length
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        optimizer = torch.optim.AdamW(self._model.parameters(), lr=lr)
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )

        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

        os.makedirs(output_dir, exist_ok=True)

        for epoch in range(epochs):
            self._model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self._device)
                attention_mask = batch["attention_mask"].to(self._device)
                labels = batch["labels"].to(self._device)

                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                num_batches += 1

            avg_train_loss = epoch_loss / max(num_batches, 1)
            history["train_loss"].append(avg_train_loss)

            # 验证
            if val_texts and val_labels:
                val_loss, val_acc = self._evaluate(val_texts, val_labels, batch_size)
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_acc)

            # 保存检查点
            ckpt_path = os.path.join(output_dir, f"epoch_{epoch+1}.pt")
            torch.save(self._model.state_dict(), ckpt_path)

        return history

    def _evaluate(
        self, texts: List[str], labels: List[str], batch_size: int = 16
    ) -> Tuple[float, float]:
        """验证集评估"""
        self._model.eval()
        label_ids = [LEVEL_TO_ID[l] for l in labels]
        dataset = DocumentDataset(texts, label_ids, self._tokenizer, self.max_length)
        loader = DataLoader(dataset, batch_size=batch_size)

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self._device)
                attention_mask = batch["attention_mask"].to(self._device)
                batch_labels = batch["labels"].to(self._device)

                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=batch_labels,
                )
                total_loss += outputs.loss.item()
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == batch_labels).sum().item()
                total += len(batch_labels)

        avg_loss = total_loss / max(len(loader), 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    def grade(self, content: str) -> DeBERTaGradingResult:
        """分级单个文档

        Args:
            content: 文档文本内容

        Returns:
            分级结果
        """
        if not _HAS_TORCH or not _HAS_TRANSFORMERS:
            return self._mock_grade(content)

        self._load_model()
        self._model.eval()

        encoding = self._tokenizer(
            content,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**encoding)
            logits = outputs.logits.cpu().numpy()[0]

        probs = self._softmax(logits)
        pred_id = int(np.argmax(probs))
        level = ID_TO_LEVEL[pred_id]
        confidence = float(probs[pred_id])

        uncertainty = float(-np.sum(probs * np.log(probs + 1e-8)))

        return DeBERTaGradingResult(
            level=level,
            confidence=confidence,
            uncertainty=uncertainty,
            reason=f"DeBERTa classification: {level} (p={confidence:.3f})",
            all_predictions=[level],
            logits=logits,
        )

    def grade_batch(self, contents: List[str]) -> List[DeBERTaGradingResult]:
        """批量分级"""
        return [self.grade(c) for c in contents]

    def _mock_grade(self, content: str) -> DeBERTaGradingResult:
        """无模型时的 mock 分级（用于测试）"""
        content_lower = content.lower()
        if any(kw in content_lower for kw in ["secret", "绝密", "密钥", "key"]):
            level, conf = "L4", 0.85
        elif any(kw in content_lower for kw in ["confidential", "机密", "财务", "salary"]):
            level, conf = "L3", 0.78
        elif any(kw in content_lower for kw in ["internal", "内部", "会议"]):
            level, conf = "L2", 0.72
        else:
            level, conf = "L1", 0.65

        return DeBERTaGradingResult(
            level=level,
            confidence=conf,
            reason=f"DeBERTa mock: {level}",
            all_predictions=[level],
        )

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / (e.sum() + 1e-8)
