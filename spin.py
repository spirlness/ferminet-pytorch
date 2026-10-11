"""
自旋投影与自旋适配 (Spin-adapted) 波函数算符模块。

在波函数输出前施加空间投影算符消除自旋污染:
1. 针对双电子单重态 (Singlet, S=0):
   通过空间置换对称化 P_{S=0} = 1/2 * (I + P_{12}) 严格消除自旋污染。
2. 针对双电子三重态 (Triplet, S=1, M_S=0):
   通过空间置换反对称化 P_{S=1} = 1/2 * (I - P_{12}) 严格保留三重态。
3. 针对闭壳层等自旋多电子体系 (n_up == n_down):
   通过上下自旋空间反转对称化 1/2 * (I + P_{up <-> down}) 抑制非物理自旋污染。
4. 提供数值极其稳定的带符号对数求和 (Signed Log-Sum-Exp) 机制，
   保证反向传播与局域能量二阶导数的数值鲁棒性。
"""

from typing import Callable, List, Optional, Tuple, Union
import torch
import torch.nn as nn


class SpinProjector(nn.Module):
    """
    自旋投影算符模块。

    针对给定自旋通道与模式，提供空间置换投影:
        Psi_proj(r) = sum_k w_k * Psi(P_k r)
    """

    def __init__(
        self,
        n_up: int,
        n_down: int,
        mode: str = "singlet",
        custom_permutations: Optional[List[Tuple[List[int], float]]] = None,
    ) -> None:
        """
        初始化自旋投影算符。

        Args:
            n_up: 自旋向上电子数
            n_down: 自旋向下电子数
            mode: 投影模式:
                - "singlet": 单重态空间对称化 (n_up=1, n_down=1 时为 1/2(I + P_12))
                - "triplet": 三重态空间反对称化 (n_up=1, n_down=1 时为 1/2(I - P_12))
                - "none": 不做投影 (恒等变换)
                - "custom": 用户自定义置换与权重
            custom_permutations: 自定义置换列表, 每项为 (permutation_indices, weight)
        """
        super().__init__()
        self.n_up = n_up
        self.n_down = n_down
        self.total_electrons = n_up + n_down
        self.mode = mode.lower()

        perms, weights = self._build_permutations(custom_permutations)
        # perms: (K, N), weights: (K,)
        self.register_buffer("perms", perms)
        self.register_buffer("weights", weights)

    def _build_permutations(
        self,
        custom_permutations: Optional[List[Tuple[List[int], float]]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """构建置换索引矩阵与对应线性组合权重。"""
        n_tot = self.total_electrons
        identity = list(range(n_tot))

        if self.mode == "none":
            # 仅恒等变换
            perms = [identity]
            weights = [1.0]

        elif self.mode == "custom":
            if not custom_permutations:
                raise ValueError("mode='custom' 时必须提供 custom_permutations 列表")
            perms = []
            weights = []
            for p, w in custom_permutations:
                if len(p) != n_tot or sorted(p) != identity:
                    raise ValueError(f"无效的置换列表: {p}，长度须为 {n_tot} 且包含 0 到 {n_tot-1}")
                perms.append(p)
                weights.append(float(w))

        elif self.mode == "singlet":
            if self.n_up == 1 and self.n_down == 1:
                # 标准双电子单重态空间对称化: 1/2 * (I + P_12)
                p12 = [1, 0]
                perms = [identity, p12]
                weights = [0.5, 0.5]
            elif self.n_up == self.n_down and self.n_up > 1:
                # 闭壳层等自旋体系: 自旋空间翻转置换 P_{up <-> down}
                # 前 n_up 个与后 n_down 个互换。
                # 对应全波函数自旋翻转宇称与空间置换奇偶性 (-1)^{n_up} 结合，
                # 保证单重态空间波函数的对称性:
                p_flip = list(range(self.n_up, n_tot)) + list(range(0, self.n_up))
                perms = [identity, p_flip]
                weights = [0.5, 0.5]
            else:
                # 其他构型默认仅恒等
                perms = [identity]
                weights = [1.0]

        elif self.mode == "triplet":
            if self.n_up == 1 and self.n_down == 1:
                # 标准双电子三重态空间反对称化: 1/2 * (I - P_12)
                p12 = [1, 0]
                perms = [identity, p12]
                weights = [0.5, -0.5]
            elif self.n_up == self.n_down and self.n_up > 1:
                # 闭壳层等自旋体系三重态空间反对称化: 1/2 * (I - P_{up <-> down})
                p_flip = list(range(self.n_up, n_tot)) + list(range(0, self.n_up))
                perms = [identity, p_flip]
                weights = [0.5, -0.5]
            else:
                perms = [identity]
                weights = [1.0]

        else:
            raise ValueError(f"不支持的投影模式: {self.mode}，可选 'singlet', 'triplet', 'custom', 'none'")

        perms_tensor = torch.tensor(perms, dtype=torch.long)
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        return perms_tensor, weights_tensor

    @property
    def num_permutations(self) -> int:
        """置换项数量 K。"""
        return self.perms.shape[0]

    def project_log_psi(
        self,
        base_forward_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        r: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对基础对数波函数施行自旋空间投影。

        Args:
            base_forward_fn: 基础波函数计算函数, 输入 r ∈ (B, N, 3),
                             返回 (log_abs_psi, sign_psi), 各为 (B,)
            r: 电子坐标, Shape: (B, N, 3)

        Returns:
            log_abs_psi_proj: 投影后的 log|Psi(r)|, Shape: (B,)
            sign_psi_proj: 投影后的 sign(Psi(r)), Shape: (B,)
        """
        # 若仅有 1 个恒等置换且权重为 1.0，直接短路以提高速度
        if self.num_permutations == 1 and abs(self.weights[0].item() - 1.0) < 1e-7:
            return base_forward_fn(r)

        batch_size = r.shape[0]
        num_k = self.num_permutations

        # 收集每一项的 log|Psi| 与 sign
        log_abs_list: List[torch.Tensor] = []
        sign_list: List[torch.Tensor] = []

        # 遍历所有置换 (对于双电子对称化 K=2)
        for k in range(num_k):
            perm = self.perms[k]
            w = self.weights[k].to(device=r.device, dtype=r.dtype)
            w_sign = torch.sign(w)
            w_log_abs = torch.log(torch.abs(w))

            # 应用置换: r_perm[:, i, :] = r[:, perm[i], :]
            r_perm = r[:, perm, :]

            # 计算基础波函数
            log_abs_k, sign_k = base_forward_fn(r_perm)

            # 将权重吸纳至符号与对数模中
            combined_sign = sign_k * w_sign
            combined_log_abs = log_abs_k + w_log_abs

            log_abs_list.append(combined_log_abs)
            sign_list.append(combined_sign)

        # 堆叠所有置换项: Shape (B, K)
        log_abs_all = torch.stack(log_abs_list, dim=1)  # (B, K)
        sign_all = torch.stack(sign_list, dim=1)        # (B, K)

        # 数值稳定的带符号 LogSumExp 合成:
        # Psi_total = sum_k sign_k * exp(log_abs_k)
        max_log = torch.amax(log_abs_all.detach(), dim=1, keepdim=True)  # (B, 1)
        all_neginf = torch.isneginf(max_log)
        safe_max_log = torch.where(all_neginf, torch.zeros_like(max_log), max_log)

        # 缩放至 [-1, 1] 附近求和
        scaled_terms = sign_all * torch.exp(log_abs_all - safe_max_log)
        scaled_terms = torch.where(
            all_neginf.expand_as(scaled_terms), torch.zeros_like(scaled_terms), scaled_terms
        )
        summed = torch.sum(scaled_terms, dim=1)  # (B,)

        abs_summed = torch.abs(summed)
        zero_mask = (abs_summed == 0) | all_neginf.squeeze(1)

        sign_proj = torch.sign(summed)
        sign_proj = torch.where(zero_mask, torch.zeros_like(sign_proj), sign_proj)

        safe_abs_summed = torch.where(zero_mask, torch.ones_like(abs_summed), abs_summed)
        log_abs_proj = torch.where(
            zero_mask,
            torch.full_like(summed, float("-inf")),
            safe_max_log.squeeze(1) + torch.log(safe_abs_summed),
        )

        return log_abs_proj, sign_proj
