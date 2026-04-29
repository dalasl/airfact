"""
双支路交叉注意力融合编码器

融合异构三维特征为统一表示：
- 左支路：高维连续特征（内容语义768维 + 行为模式96维 = 864维）
- 右支路：低维离散特征（权限上下文 ~32维）
- 融合：线性投影 → 交叉注意力 → LayerNorm

论文对应：
    - 公式 (eq:feature-fusion):
      z_u = LayerNorm(W_z · [CrossAttn(h_cont, h_disc); h_cont; h_disc])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CrossAttentionBlock(nn.Module):
    """交叉注意力模块

    左支路（高维连续特征）作为 Query，
    右支路（低维离散特征）作为 Key/Value，
    捕捉权限上下文对内容-行为特征的调制关系。

    Args:
        query_dim: Query 维度（左支路）
        kv_dim: Key/Value 维度（右支路）
        num_heads: 注意力头数
        dropout: Dropout 比率
    """

    def __init__(self, query_dim: int, kv_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        assert query_dim % num_heads == 0, "query_dim 必须能被 num_heads 整除"

        # 线性投影（将不同维度投影到统一空间）
        self.W_q = nn.Linear(query_dim, query_dim)
        self.W_k = nn.Linear(kv_dim, query_dim)
        self.W_v = nn.Linear(kv_dim, query_dim)
        self.W_o = nn.Linear(query_dim, query_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: 左支路特征 [B, query_dim]
            key_value: 右支路特征 [B, kv_dim]

        Returns:
            交叉注意力输出 [B, query_dim]
        """
        B = query.size(0)

        # 添加序列维度（单 token 序列）
        query = query.unsqueeze(1)      # [B, 1, query_dim]
        key_value = key_value.unsqueeze(1)  # [B, 1, kv_dim]

        Q = self.W_q(query)             # [B, 1, query_dim]
        K = self.W_k(key_value)         # [B, 1, query_dim]
        V = self.W_v(key_value)         # [B, 1, query_dim]

        # 多头拆分
        Q = Q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力计算
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, 1, -1)

        output = self.W_o(attn_output).squeeze(1)  # [B, query_dim]
        return output


class DualBranchFusionEncoder(nn.Module):
    """双支路交叉注意力融合编码器

    将三维异构特征融合为统一用户表示向量 z_u。

    架构：
        左支路 h_cont (864d) ─┐
                              ├─→ CrossAttention ─→ Concat ─→ W_z ─→ LayerNorm ─→ z_u
        右支路 h_disc (32d)  ─┘

    公式：z_u = LayerNorm(W_z · [CrossAttn(h_cont, h_disc); h_cont; h_disc])

    Args:
        left_dim: 左支路输入维度（内容语义768 + 行为模式96 = 864）
        right_dim: 右支路输入维度（权限上下文 ~32）
        hidden_dim: 隐藏层维度
        num_heads: 交叉注意力头数
        dropout: Dropout 比率
    """

    def __init__(
        self,
        left_dim: int = 864,
        right_dim: int = 32,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 左支路线性投影（降维至统一空间）
        self.left_proj = nn.Sequential(
            nn.Linear(left_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 右支路线性投影
        self.right_proj = nn.Sequential(
            nn.Linear(right_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 交叉注意力
        self.cross_attention = CrossAttentionBlock(
            query_dim=hidden_dim,
            kv_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # 最终融合投影 W_z
        # 输入 = CrossAttn输出(hidden_dim) + h_cont(hidden_dim) + h_disc(hidden_dim)
        concat_dim = hidden_dim * 3
        self.W_z = nn.Linear(concat_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, content_behavior: torch.Tensor, permission: torch.Tensor
    ) -> torch.Tensor:
        """前向传播

        Args:
            content_behavior: 内容语义 + 行为模式拼接 [B, 864]
            permission: 权限上下文 [B, 32]

        Returns:
            融合用户表示向量 z_u [B, hidden_dim]
        """
        # 分支投影
        h_cont = self.left_proj(content_behavior)   # [B, hidden_dim]
        h_disc = self.right_proj(permission)         # [B, hidden_dim]

        # 交叉注意力
        cross_out = self.cross_attention(h_cont, h_disc)  # [B, hidden_dim]

        # 拼接与融合
        concat = torch.cat([cross_out, h_cont, h_disc], dim=-1)  # [B, 3*hidden_dim]
        z_u = self.W_z(concat)
        z_u = self.layer_norm(z_u)

        return z_u

    def encode_user(
        self,
        content_semantic: torch.Tensor,
        behavior_pattern: torch.Tensor,
        permission_context: torch.Tensor,
    ) -> torch.Tensor:
        """编码完整三维用户特征

        Args:
            content_semantic: 内容语义特征 [B, 768]
            behavior_pattern: 行为模式特征 [B, 96]
            permission_context: 权限上下文特征 [B, 32]

        Returns:
            融合用户表示向量 z_u [B, hidden_dim]
        """
        content_behavior = torch.cat([content_semantic, behavior_pattern], dim=-1)
        return self.forward(content_behavior, permission_context)
