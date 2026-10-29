"""
ferminet_pytorch.psiformer_attention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Psiformer 风格的坐标几何偏置与自旋嵌入多头自注意力层，以注意力替代持续维护的双电子隐藏流；仍有成对几何中间张量。
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PsiformerAttentionBlock(nn.Module):
    """
    坐标感知与自旋嵌入多头自注意力等变交互块 (修复 BUG-04)。

    架构特性:
    1. 彻底淘汰显式双电子张量流 (B, N, N, D_2)；
    2. 注入可学习自旋嵌入 (Spin Embedding)，使注意力机制能够感知同自旋与异自旋电子对，
       保障开壳层自由基与多重态体系的物理等变与自旋保真度；
    3. 将电子间相对坐标与平滑距离 [r_ij, d_ij^2/(1+d_ij)] 映射为注意力矩阵的空间几何偏置；
    4. 显存复杂度降低至 O(N^2 * num_heads)，实际可扩展规模须经基准验证。
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        mlp_ratio: int = 2,
        n_up: Optional[int] = None,
        n_down: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)
        self.n_up = n_up
        self.n_down = n_down

        # 自旋类型嵌入: 区分向上自旋 (0) 与向下自旋 (1) 电子 (彻底修复 BUG-04)
        self.spin_embed = nn.Embedding(2, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)

        # 几何空间偏置映射 MLP: [r_ij (3), d_ij (1)] -> num_heads
        bias_hidden = max(1, embed_dim // 2)
        self.geom_bias_mlp = nn.Sequential(
            nn.Linear(4, bias_hidden),
            nn.Tanh(),
            nn.Linear(bias_hidden, num_heads),
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_ratio * embed_dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * embed_dim, embed_dim),
        )

    def forward(
        self,
        h: torch.Tensor,
        r_ee: Optional[torch.Tensor] = None,
        d_ee: Optional[torch.Tensor] = None,
        spin_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            h: 单电子隐藏状态, shape (B, N, embed_dim), dtype torch.float32/float64
            r_ee: 电子对相对位移 (来自 GeometricContext), shape (B, N, N, 3) 或 None
            d_ee: 电子对平滑欧氏距离 (来自 GeometricContext), shape (B, N, N, 1) 或 None
            spin_ids: 可选的电子自旋类型张量 (0 表示向上自旋, 1 表示向下自旋), shape (N,) 或 (B, N)

        Returns:
            更新后的单电子隐藏状态, shape (B, N, embed_dim)
        """
        B, N, C = h.shape

        # 0. 注入自旋嵌入 (Spin Embedding): 赋予自注意力感知同自旋/异自旋极性的物理能力
        if spin_ids is not None:
            h_input = h + self.spin_embed(spin_ids)
        elif self.n_up is not None and self.n_down is not None:
            # 自动根据 n_up 与 n_down 构建自旋索引张量: [0, ..., 0, 1, ..., 1]
            s_idx = torch.zeros(N, dtype=torch.long, device=h.device)
            s_idx[self.n_up:] = 1
            h_input = h + self.spin_embed(s_idx).unsqueeze(0)
        else:
            h_input = h

        # 1. Pre-LayerNorm 与 QKV 投影
        norm_h = self.norm1(h_input)
        qkv = self.qkv_proj(norm_h).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # 各为 (B, N, num_heads, head_dim)

        # 2. 变换维度用于批量矩阵乘法 (B, num_heads, N, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # 点积自注意力得分: (B, num_heads, N, N)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # 3. 注入相对坐标空间几何偏置
        if r_ee is not None and d_ee is not None:
            geom_feat = torch.cat([r_ee, d_ee.square() / (1.0 + d_ee)], dim=-1)  # (B, N, N, 4)
            geom_bias = self.geom_bias_mlp(geom_feat).permute(0, 3, 1, 2)  # (B, num_heads, N, N)
            attn_scores = attn_scores + geom_bias

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)  # (B, num_heads, N, head_dim)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N, C)

        # 4. 残差连接与 FFN
        h = h + self.out_proj(attn_out)
        h = h + self.ffn(self.norm2(h))
        return h
