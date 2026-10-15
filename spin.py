"""
自旋投影与自旋适配 (Spin-adapted) 波函数算符模块。

在波函数输出前施加空间投影算符消除自旋污染:
1. 针对双电子单重态 (Singlet, S=0):
   通过空间置换对称化 P_{S=0} = 1/2 * (I + P_{12}) 严格消除自旋污染。
2. 针对双电子三重态 (Triplet, S=1, M_S=0):
   通过空间置换反对称化 P_{S=1} = 1/2 * (I - P_{12}) 严格保留三重态。
3. 多电子体系通过 S^2 的谱投影多项式选择目标自旋，
   而非仅固定整体自旋翻转宇称。输入须在各自旋组内反对称。
4. 提供数值极其稳定的带符号对数求和 (Signed Log-Sum-Exp) 机制，
   保证反向传播与局域能量二阶导数的数值鲁棒性。
"""

from collections import defaultdict
from fractions import Fraction
from typing import Callable, List, Optional, Tuple
import torch
import torch.nn as nn


class SpinProjector(nn.Module):
    """
    自旋投影算符模块。

    针对给定自旋通道与模式，提供空间置换投影:
        Psi_proj(r) = sum_k w_k * Psi(P_k r)

    singlet/triplet 模式要求基础波函数在各自旋组内反对称，并要求目标
    自旋与固定 M_S 相容。多电子投影最多需要 binom(N, n_up) 次前向计算。
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
        if n_up < 0 or n_down < 0 or n_up + n_down == 0:
            raise ValueError("Electron counts must be nonnegative with at least one electron")
        self.n_up = n_up
        self.n_down = n_down
        self.total_electrons = n_up + n_down
        self.mode = mode.lower()

        perms, weights = self._build_permutations(custom_permutations)
        # perms: (K, N), weights: (K,)
        self.register_buffer("perms", perms)
        self.register_buffer("weights", weights)

    def _spin_eigenprojector(self, target: int) -> Tuple[List[List[int]], List[float]]:
        """Expand product_{s != target} (S^2-s(s+1))/(target(target+1)-s(s+1)).

        In a fixed M sector, S^2 = N/2 + M^2 - sum_{up,down} P_ij.
        See https://arxiv.org/html/2406.01222v1#S1.SS1, Eq. (1).
        Same-spin antisymmetry reduces the expansion to at most binom(N, n_up)
        coordinate assignments. Construction uses exact rational coefficients.
        """
        n = self.total_electrons
        minimum = abs(self.n_up - self.n_down) // 2
        if n % 2 or not minimum <= target <= n // 2:
            raise ValueError(f"mode={self.mode!r} is incompatible with n_up={self.n_up}, n_down={self.n_down}")
        constant = Fraction(n, 2) + Fraction((self.n_up - self.n_down) ** 2, 4)
        terms = {tuple(range(n)): Fraction(1)}
        for spin in range(minimum, n // 2 + 1):
            if spin == target:
                continue
            eigenvalue = spin * (spin + 1)
            denominator = target * (target + 1) - eigenvalue
            updated = defaultdict(Fraction)
            for perm, weight in terms.items():
                updated[perm] += weight * (constant - eigenvalue) / denominator
                for i in range(self.n_up):
                    for j in range(self.n_up, n):
                        swapped = [j if k == i else i if k == j else k for k in perm]
                        up, down = swapped[:self.n_up], swapped[self.n_up:]
                        inversions = sum(
                            group[a] > group[b] for group in (up, down)
                            for a in range(len(group)) for b in range(a + 1, len(group))
                        )
                        canonical = tuple(sorted(up) + sorted(down))
                        updated[canonical] -= weight * ((-1) ** inversions) / denominator
            terms = {perm: weight for perm, weight in updated.items() if weight}
        return [list(perm) for perm in terms], [float(weight) for weight in terms.values()]

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

        elif self.mode in ("singlet", "triplet"):
            perms, weights = self._spin_eigenprojector(0 if self.mode == "singlet" else 1)

        else:
            raise ValueError(f"不支持的投影模式: {self.mode}，可选 'singlet', 'triplet', 'custom', 'none'")

        perms_tensor = torch.tensor(perms, dtype=torch.long)
        weights_tensor = torch.tensor(weights, dtype=torch.float64)
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
        # 单项单位权重可省略求和，但仍必须执行所指定的置换。
        if self.num_permutations == 1 and self.weights[0].item() == 1.0:
            return base_forward_fn(r[:, self.perms[0], :])

        batch_size = r.shape[0]
        num_k = self.num_permutations
        num_electrons = r.shape[1]

        # 批量向量化广播置换: Shape (B, K, N, 3)
        perms_dev = self.perms.to(device=r.device)
        r_all_perms = r[:, perms_dev, :]

        # 展平为大批次单次调用基础波函数前向: Shape (B * K, N, 3)
        r_batched = r_all_perms.reshape(batch_size * num_k, num_electrons, 3)
        log_abs_batched, sign_batched = base_forward_fn(r_batched)

        # 恢复 (B, K) 形状
        log_abs_all = log_abs_batched.reshape(batch_size, num_k)
        sign_all = sign_batched.reshape(batch_size, num_k)

        # 注入置换权重系数: Shape (1, K)
        weights_dev = self.weights.to(device=r.device, dtype=r.dtype)
        w_sign = torch.sign(weights_dev).unsqueeze(0)
        w_log_abs = torch.log(torch.abs(weights_dev)).unsqueeze(0)

        log_abs_all = log_abs_all + w_log_abs
        sign_all = sign_all * w_sign

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

        safe_abs_summed = torch.clamp(abs_summed, min=1e-20)
        log_abs_proj = torch.where(
            zero_mask,
            torch.full_like(summed, float("-inf")),
            safe_max_log.squeeze(1) + torch.log(safe_abs_summed),
        )

        return log_abs_proj, sign_proj


# 别名兼容
SpinProjectedWaveFunction = SpinProjector

