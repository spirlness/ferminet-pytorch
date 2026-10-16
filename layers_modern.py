"""
ferminet_pytorch.layers_modern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
消除 masked_fill 与全尺寸 torch.cat 的零冗余等变网络交互层。
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


class FermiInteractionLayerModern(nn.Module):
    """
    现代化自旋等变相互作用层。

    优化机理:
    1. 消除 masked_fill: 采用全量求和后原地扣减对角线元素，单层节省 (B, N, N, D_2) 副本；
    2. 消除 expand + torch.cat: 利用线性映射可加性分块投影均值特征，通过广播直接相加；
    3. 预注册通道维度归一化分母与有效布尔掩码 Buffer: 消除动态掩码与标量分支，
       保证 n_up=1 或 n_down=1（如 H2 分子）时异自旋均值池化物理保真，实现纯张量无分支（Branchless）运算。
    """
    def __init__(
        self,
        d_in_single: int,
        d_in_double: int,
        d_out_single: int,
        d_out_double: int,
        n_up: int,
        n_down: int,
        activation: nn.Module = nn.Tanh(),
    ) -> None:
        super().__init__()
        self.d_in_single = d_in_single
        self.d_in_double = d_in_double
        self.d_out_single = d_out_single
        self.d_out_double = d_out_double
        self.n_up = n_up
        self.n_down = n_down
        self.total_electrons = n_up + n_down
        self.act = activation

        # 1. 单电子流分块投影层 (取代大矩阵拼接后的单一线性层)
        self.proj_h1_self = nn.Linear(d_in_single, d_out_single)
        self.proj_h1_mean_up = nn.Linear(d_in_single, d_out_single, bias=False) if n_up > 0 else None
        self.proj_h1_mean_down = nn.Linear(d_in_single, d_out_single, bias=False) if n_down > 0 else None
        self.proj_h2_mean_up = nn.Linear(d_in_double, d_out_single, bias=False) if n_up > 0 else None
        self.proj_h2_mean_down = nn.Linear(d_in_double, d_out_single, bias=False) if n_down > 0 else None
        self.res_proj_single = (
            nn.Linear(d_in_single, d_out_single, bias=False)
            if d_in_single != d_out_single else nn.Identity()
        )

        # 2. 双电子流投影层
        self.linear_double = nn.Linear(d_in_double, d_out_double)
        self.res_proj_double = (
            nn.Linear(d_in_double, d_out_double, bias=False)
            if d_in_double != d_out_double else nn.Identity()
        )

        # 3. 预注册通道维度的归一化分母与有效掩码 Buffer (消除动态掩码与标量分支，彻底修复 BUG-01)
        # 电子 i 属于向上自旋时，排除自身后的向上计数为 n_up - 1；电子 i 属于向下自旋时，向上计数为 n_up
        counts_up = torch.full((1, self.total_electrons, 1), float(n_up))
        valid_mask_up = torch.ones((1, self.total_electrons, 1), dtype=torch.bool)
        if n_up > 0:
            counts_up[:, :n_up, :] = float(n_up - 1)
            if n_up == 1:
                valid_mask_up[:, :n_up, :] = False
        else:
            valid_mask_up[:] = False
        self.register_buffer("denom_up", counts_up.clamp(min=1.0))
        self.register_buffer("valid_mask_up", valid_mask_up)

        counts_down = torch.full((1, self.total_electrons, 1), float(n_down))
        valid_mask_down = torch.ones((1, self.total_electrons, 1), dtype=torch.bool)
        if n_down > 0:
            counts_down[:, n_up:, :] = float(n_down - 1)
            if n_down == 1:
                valid_mask_down[:, n_up:, :] = False
        else:
            valid_mask_down[:] = False
        self.register_buffer("denom_down", counts_down.clamp(min=1.0))
        self.register_buffer("valid_mask_down", valid_mask_down)

    def forward(
        self,
        h1: torch.Tensor,
        h2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            h1: 单电子特征, shape: (B, N, d_in_single), dtype: torch.float32/float64
            h2: 双电子特征, shape: (B, N, N, d_in_double), dtype: torch.float32/float64

        Returns:
            Tuple[h1_next, h2_next]:
                h1_next: shape (B, N, d_out_single)
                h2_next: shape (B, N, N, d_out_double) 或 None
        """
        B, N, _ = h1.shape

        # --- 1. 双电子流池化：就地消除对角线，0 额外全矩阵分配 ---
        if h2 is not None:
            # 对自旋向上电子求和
            if self.n_up > 0:
                sum_h2_up = torch.sum(h2[:, :, :self.n_up, :], dim=2)  # (B, N, D_2)
                diag_up = torch.arange(self.n_up, device=h2.device)
                sum_h2_up[:, diag_up, :] = sum_h2_up[:, diag_up, :] - h2[:, diag_up, diag_up, :]
                mean_h2_up = torch.where(
                    self.valid_mask_up,
                    sum_h2_up / self.denom_up,
                    torch.zeros_like(sum_h2_up),
                )
            else:
                mean_h2_up = torch.zeros(B, N, self.d_in_double, device=h1.device, dtype=h1.dtype)

            # 对自旋向下电子求和
            if self.n_down > 0:
                sum_h2_down = torch.sum(h2[:, :, self.n_up:, :], dim=2)  # (B, N, D_2)
                diag_down = torch.arange(self.n_down, device=h2.device) + self.n_up
                sum_h2_down[:, diag_down, :] = sum_h2_down[:, diag_down, :] - h2[:, diag_down, diag_down, :]
                mean_h2_down = torch.where(
                    self.valid_mask_down,
                    sum_h2_down / self.denom_down,
                    torch.zeros_like(sum_h2_down),
                )
            else:
                mean_h2_down = torch.zeros(B, N, self.d_in_double, device=h1.device, dtype=h1.dtype)
        else:
            mean_h2_up = None
            mean_h2_down = None

        # --- 2. 单电子均值特征投影 ---
        h1_out = self.proj_h1_self(h1)  # (B, N, D_out_1)
        if self.proj_h1_mean_up is not None:
            mean_h1_up = torch.mean(h1[:, :self.n_up, :], dim=1, keepdim=True)  # (B, 1, D_1)
            h1_out = h1_out + self.proj_h1_mean_up(mean_h1_up)  # 自动广播累加

        if self.proj_h1_mean_down is not None:
            mean_h1_down = torch.mean(h1[:, self.n_up:, :], dim=1, keepdim=True)  # (B, 1, D_1)
            h1_out = h1_out + self.proj_h1_mean_down(mean_h1_down)

        if self.proj_h2_mean_up is not None and mean_h2_up is not None:
            h1_out = h1_out + self.proj_h2_mean_up(mean_h2_up)

        if self.proj_h2_mean_down is not None and mean_h2_down is not None:
            h1_out = h1_out + self.proj_h2_mean_down(mean_h2_down)

        h1_next = self.act(h1_out) + self.res_proj_single(h1)

        # --- 3. 双电子流前向更新 ---
        if h2 is not None:
            h2_transformed = self.act(self.linear_double(h2))
            h2_next = h2_transformed + self.res_proj_double(h2)
        else:
            h2_next = None

        return h1_next, h2_next
