# U-Net 去噪训练配对数据

本目录存放 U-Net 去噪模型的训练数据（干净-噪声图像配对），因体积过大（约 2.4GB）未纳入版本控制。

## 重建方法

```bash
python src/data_pipeline/noise_augmentor.py \
    --clean-dir data/rvlcdip \
    --output-dir data/unet_pairs \
    --seed 42 \
    --augments 3
```

**前置条件：** `data/rvlcdip/` 目录已包含 3,000 张 TIF 图像。

## 预期输出

```
data/unet_pairs/
├── clean/     3,000 张干净 PNG（从 TIF 转换）
├── noisy/     9,000 张噪声 PNG（每张 × 3 种噪声）
└── pairs.csv  9,001 行（header + 9,000 条配对记录）
```

## 噪声类型

| 类型 | 数量 | 模拟场景 |
|------|------|----------|
| gaussian | 3,000 | 扫描仪传感器噪声 |
| salt_pepper | 3,000 | 传输损坏 |
| jpeg | 3,000 | JPEG 压缩伪影 |

## 参数

- 随机种子：42（确保可复现）
- 高斯噪声 σ：10~50
- 椒盐噪声密度：1%~5%
- JPEG 质量：10~40
