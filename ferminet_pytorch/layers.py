"""Spin-equivariant interactions with one parameterization and static pooling."""

import torch
from torch import nn
from torch.nn import functional as F


class FermiInteractionLayer(nn.Module):
    """Two streams with self-excluded, spin-separated electron means.

    Linear weights are stored once; their feature blocks are applied separately
    so broadcasting does not require a concatenated aggregation tensor.
    """

    def __init__(
        self,
        in_dim_one,
        in_dim_two,
        out_dim_one,
        out_dim_two,
        n_up,
        n_down,
        produce_two_electron_output=True,
    ):
        super().__init__()
        if min(n_up, n_down) < 0 or n_up + n_down == 0:
            raise ValueError("Electron counts must be nonnegative and nonempty")
        self.in_dim_one, self.in_dim_two = in_dim_one, in_dim_two
        self.n_up, self.n_down = n_up, n_down
        self.produce_two_electron_output = produce_two_electron_output
        self.linear_one = nn.Linear(3 * in_dim_one + 2 * in_dim_two, out_dim_one)
        self.res_proj_one = (
            nn.Linear(in_dim_one, out_dim_one, bias=False)
            if in_dim_one != out_dim_one
            else nn.Identity()
        )
        self.act = nn.Tanh()
        if produce_two_electron_output:
            self.linear_two = nn.Linear(in_dim_two, out_dim_two)
            self.res_proj_two = (
                nn.Linear(in_dim_two, out_dim_two, bias=False)
                if in_dim_two != out_dim_two
                else nn.Identity()
            )
        n = n_up + n_down
        for name, start, stop in (("up", 0, n_up), ("down", n_up, n)):
            weights = torch.tensor(
                [
                    [float(start <= j < stop and i != j) for j in range(n)]
                    for i in range(n)
                ]
            )
            self.register_buffer(
                "denom_" + name,
                weights.sum(-1).clamp_min(1)[None, :, None],
                persistent=False,
            )
            self.register_buffer("pool_" + name, weights, persistent=False)

    def forward(self, h1, h2):
        n, d = h1.shape[1:]
        w_self, w_up, w_down, w_pair_up, w_pair_down = self.linear_one.weight.split(
            [d, d, d, self.in_dim_two, self.in_dim_two], dim=1
        )
        out = F.linear(h1, w_self, self.linear_one.bias)
        if self.n_up:
            out = out + F.linear(h1[:, : self.n_up].mean(1, keepdim=True), w_up)
            out = out + F.linear(
                torch.einsum("ij,bijf->bif", self.pool_up, h2) / self.denom_up,
                w_pair_up,
            )
        if self.n_down:
            out = out + F.linear(h1[:, self.n_up :].mean(1, keepdim=True), w_down)
            out = out + F.linear(
                torch.einsum("ij,bijf->bif", self.pool_down, h2) / self.denom_down,
                w_pair_down,
            )
        single = self.act(out) + self.res_proj_one(h1)
        pair = (
            self.act(self.linear_two(h2)) + self.res_proj_two(h2)
            if self.produce_two_electron_output
            else None
        )
        return single, pair
