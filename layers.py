from typing import Optional, Tuple
import torch
import torch.nn as nn


class FermiInteractionLayer(nn.Module):
    """
    FermiNet 自旋等变相互作用层 (FermiNet Interaction / Equivariant Layer)。

    针对单电子流 (One-electron stream) 和双电子流 (Two-electron stream) 进行特征交互与更新。
    单电子特征聚合操作分为自旋向上电子集合 (Up-spin) 与自旋向下电子集合 (Down-spin)，
    保证模型网络关于同自旋电子置换的等变性 (Permutation Equivariance)。

    注意：当 ``produce_two_electron_output=False`` 时（FermiNet 装配中的末层），
    双电子流更新网络 (`linear_two` / `res_proj_two`) 不会被构造——末层的双电子流
    输出不被任何后续模块消费（轨道矩阵仅由单电子流 h1 构造），若仍构造会得到
    恒收不到梯度的空转参数。此时 ``forward`` 返回 ``(h1_next, None)``，
    而双电子流 h2 输入仍参与单电子流的均值池化聚合。
    """

    def __init__(
        self,
        in_dim_one: int,
        in_dim_two: int,
        out_dim_one: int,
        out_dim_two: int,
        produce_two_electron_output: bool = True,
    ) -> None:
        """
        初始化自旋等变相互作用层。

        Args:
            in_dim_one (int): 输入单电子流特征维度。
            in_dim_two (int): 输入双电子流特征维度。
            out_dim_one (int): 输出单电子流特征维度。
            out_dim_two (int): 输出双电子流特征维度。
            produce_two_electron_output (bool): 是否构造双电子流更新网络并输出
                更新后的 h2 (默认 True，保持完整层契约)。置为 False 时不创建
                `linear_two` 与 `res_proj_two`，``forward`` 返回 `(h1_next, None)`。
                用于模型装配的末层：其 h2 输出无人消费，不构造对应更新网络
                以避免空转参数。
        """
        super().__init__()
        self.in_dim_one = in_dim_one
        self.in_dim_two = in_dim_two
        self.out_dim_one = out_dim_one
        self.out_dim_two = out_dim_two
        self.produce_two_electron_output = produce_two_electron_output

        # 聚合特征拼接总维度:
        # h_i (in_dim_one) + mean(h_up) (in_dim_one) + mean(h_down) (in_dim_one)
        # + mean_up(h_ij) (in_dim_two) + mean_down(h_ij) (in_dim_two)
        agg_dim_one = 3 * in_dim_one + 2 * in_dim_two

        # 单电子流更新网络
        self.linear_one = nn.Linear(agg_dim_one, out_dim_one)
        # 双电子流更新网络 (末层不消费 h2 输出时不构造，避免空转参数)
        if self.produce_two_electron_output:
            self.linear_two = nn.Linear(in_dim_two, out_dim_two)

        self.act = nn.Tanh()

        # 残差连接对齐投影 (当输入与输出维度不一致时使用线性映射，一致时为恒等映射)
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
        前向执行自旋分离聚合与流特征更新。

        Args:
            h1 (torch.Tensor): 单电子流特征。
                - Shape: `(B, N, in_dim_one)`，其中 N = n_up + n_down。
                - Dtype: `torch.float32` 或 `torch.float64`。
            h2 (torch.Tensor): 双电子流特征。
                - Shape: `(B, N, N, in_dim_two)`。
                - Dtype: 与 h1 一致。
            n_up (int): 自旋向上电子数目 (对应前 n_up 个电子)。
            n_down (int): 自旋向下电子数目 (对应后 n_down 个电子)。

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - h1_next (torch.Tensor): 更新后的单电子流特征。
                    - Shape: `(B, N, out_dim_one)`。
                    - Dtype: 与 h1 一致。
                - h2_next (Optional[torch.Tensor]): 更新后的双电子流特征。
                    - Shape: `(B, N, N, out_dim_two)`。
                    - Dtype: 与 h2 一致。
                    - 当 ``produce_two_electron_output=False`` 时为 `None`
                      (末层装配：h2 输出无人消费，仅返回 h1_next)。
        """
        batch_size, num_electrons, _ = h1.shape
        assert num_electrons == n_up + n_down, (
            f"电子数总和不匹配: {num_electrons} != {n_up} + {n_down}"
        )

        # ---------------------------------------------------------------------
        # 1. 单电子流聚合 (Mean Pooling over spin channels)
        # ---------------------------------------------------------------------
        # 自旋向上均值: (B, 1, in_dim_one)
        if n_up > 0:
            mean_h1_up = torch.mean(h1[:, :n_up, :], dim=1, keepdim=True)
        else:
            mean_h1_up = torch.zeros(
                batch_size, 1, self.in_dim_one, device=h1.device, dtype=h1.dtype
            )

        # 自旋向下均值: (B, 1, in_dim_one)
        if n_down > 0:
            mean_h1_down = torch.mean(h1[:, n_up:, :], dim=1, keepdim=True)
        else:
            mean_h1_down = torch.zeros(
                batch_size, 1, self.in_dim_one, device=h1.device, dtype=h1.dtype
            )

        # 广播扩展至 (B, N, in_dim_one)
        mean_h1_up_expanded = mean_h1_up.expand(batch_size, num_electrons, -1)
        mean_h1_down_expanded = mean_h1_down.expand(batch_size, num_electrons, -1)

        # ---------------------------------------------------------------------
        # 2. 双电子流聚合 (Mean Pooling over spin channels along j dimension)
        # ---------------------------------------------------------------------
        # 电子 i 与所有自旋向上电子 j 的双电子特征均值: (B, N, in_dim_two)
        if n_up > 0:
            mean_h2_up = torch.mean(h2[:, :, :n_up, :], dim=2)
        else:
            mean_h2_up = torch.zeros(
                batch_size, num_electrons, self.in_dim_two, device=h2.device, dtype=h2.dtype
            )

        # 电子 i 与所有自旋向下电子 j 的双电子特征均值: (B, N, in_dim_two)
        if n_down > 0:
            mean_h2_down = torch.mean(h2[:, :, n_up:, :], dim=2)
        else:
            mean_h2_down = torch.zeros(
                batch_size, num_electrons, self.in_dim_two, device=h2.device, dtype=h2.dtype
            )

        # ---------------------------------------------------------------------
        # 3. 拼接聚合特征并更新单电子流
        # a_i = [h_i, mean_h1_up, mean_h1_down, mean_h2_up, mean_h2_down]
        # shape: (B, N, 3 * in_dim_one + 2 * in_dim_two)
        # ---------------------------------------------------------------------
        h1_agg = torch.cat(
            [h1, mean_h1_up_expanded, mean_h1_down_expanded, mean_h2_up, mean_h2_down],
            dim=-1,
        )

        h1_transformed = self.act(self.linear_one(h1_agg))
        h1_next = h1_transformed + self.res_proj_one(h1)

        # ---------------------------------------------------------------------
        # 4. 更新双电子流特征 (末层装配时不构造更新网络，直接返回 None)
        # h2_ij 经线性变换后加激活与残差连接
        # shape: (B, N, N, out_dim_two)
        # ---------------------------------------------------------------------
        if not self.produce_two_electron_output:
            return h1_next, None

        h2_transformed = self.act(self.linear_two(h2))
        h2_next = h2_transformed + self.res_proj_two(h2)

        return h1_next, h2_next
