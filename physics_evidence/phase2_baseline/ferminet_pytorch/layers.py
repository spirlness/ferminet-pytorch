"""
FermiNet 自旋等变相互作用层。

单电子流 h1 与双电子流 h2 通过自旋分离的均值池化进行特征交互，
保证同自旋电子置换等变性。
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


class FermiInteractionLayer(nn.Module):
    """
    FermiNet 自旋等变交互层。

    聚合特征: [h_i, mean(h_up), mean(h_down), mean_up(h_ij), mean_down(h_ij)]
    """

    def __init__(
        self,
        in_dim_one: int,
        in_dim_two: int,
        out_dim_one: int,
        out_dim_two: int,
        produce_two_electron_output: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim_one = in_dim_one
        self.in_dim_two = in_dim_two
        self.out_dim_one = out_dim_one
        self.out_dim_two = out_dim_two
        self.produce_two_electron_output = produce_two_electron_output

        agg_dim_one = 3 * in_dim_one + 2 * in_dim_two
        self.linear_one = nn.Linear(agg_dim_one, out_dim_one)

        if self.produce_two_electron_output:
            self.linear_two = nn.Linear(in_dim_two, out_dim_two)

        self.act = nn.Tanh()

        # 残差投影
        if in_dim_one != out_dim_one:
            self.res_proj_one = nn.Linear(in_dim_one, out_dim_one, bias=False)
        else:
            self.res_proj_one = nn.Identity()

        if self.produce_two_electron_output:
            if in_dim_two != out_dim_two:
                self.res_proj_two = nn.Linear(in_dim_two, out_dim_two, bias=False)
            else:
                self.res_proj_two = nn.Identity()

    def forward(
        self,
        h1: torch.Tensor,
        h2: torch.Tensor,
        n_up: int,
        n_down: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            h1: 单电子流, Shape: (B, N, in_dim_one)
            h2: 双电子流, Shape: (B, N, N, in_dim_two)
            n_up: 自旋向上电子数
            n_down: 自旋向下电子数

        Returns:
            h1_next: Shape: (B, N, out_dim_one)
            h2_next: Shape: (B, N, N, out_dim_two) 或 None
        """
        batch_size, num_electrons, _ = h1.shape

        # 1. 单电子流均值池化
        if n_up > 0:
            mean_h1_up = torch.mean(h1[:, :n_up, :], dim=1, keepdim=True)
        else:
            mean_h1_up = torch.zeros(
                batch_size, 1, self.in_dim_one, device=h1.device, dtype=h1.dtype
            )

        if n_down > 0:
            mean_h1_down = torch.mean(h1[:, n_up:, :], dim=1, keepdim=True)
        else:
            mean_h1_down = torch.zeros(
                batch_size, 1, self.in_dim_one, device=h1.device, dtype=h1.dtype
            )

        mean_h1_up_expanded = mean_h1_up.expand(batch_size, num_electrons, -1)
        mean_h1_down_expanded = mean_h1_down.expand(batch_size, num_electrons, -1)

        # 2. 双电子流均值池化（显式剔除自身对角线自交互项 i=j）
        eye_mask = torch.eye(num_electrons, dtype=torch.bool, device=h2.device)
        h2_no_diag = h2.masked_fill(eye_mask.unsqueeze(0).unsqueeze(-1), 0.0)

        # 2.1 自旋向上通道均值池化 (同自旋排除自身除以 n_up - 1，异自旋除以 n_up)
        sum_h2_up = torch.sum(h2_no_diag[:, :, :n_up, :], dim=2)
        counts_up = torch.full((1, num_electrons, 1), float(n_up), device=h2.device, dtype=h2.dtype)
        if n_up > 0:
            counts_up[:, :n_up, :] = float(n_up - 1)
        mean_h2_up = torch.where(
            counts_up > 0,
            sum_h2_up / counts_up.clamp(min=1.0),
            torch.zeros_like(sum_h2_up),
        )

        # 2.2 自旋向下通道均值池化 (同自旋排除自身除以 n_down - 1，异自旋除以 n_down)
        sum_h2_down = torch.sum(h2_no_diag[:, :, n_up:, :], dim=2)
        counts_down = torch.full((1, num_electrons, 1), float(n_down), device=h2.device, dtype=h2.dtype)
        if n_down > 0:
            counts_down[:, n_up:, :] = float(n_down - 1)
        mean_h2_down = torch.where(
            counts_down > 0,
            sum_h2_down / counts_down.clamp(min=1.0),
            torch.zeros_like(sum_h2_down),
        )


        # 3. 拼接并更新单电子流
        h1_agg = torch.cat(
            [h1, mean_h1_up_expanded, mean_h1_down_expanded, mean_h2_up, mean_h2_down],
            dim=-1,
        )
        h1_transformed = self.act(self.linear_one(h1_agg))
        h1_next = h1_transformed + self.res_proj_one(h1)

        # 4. 更新双电子流
        if not self.produce_two_electron_output:
            return h1_next, None

        h2_transformed = self.act(self.linear_two(h2))
        h2_next = h2_transformed + self.res_proj_two(h2)

        return h1_next, h2_next


# 别名兼容与现代化组件导出
FermiNetLayer = FermiInteractionLayer

from .layers_modern import FermiInteractionLayerModern


