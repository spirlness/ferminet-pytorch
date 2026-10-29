"""Shared signed determinant sums with exact singular derivatives.

The cofactor rule differentiates minors only to the requested derivative order;
it does not recursively expand every determinant to scalars in the forward pass.
Log scales are numerical constants, intentionally detached from differentiation.
"""

import math
from typing import Optional, Tuple

import torch


def _cofactors(matrix: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
    size = matrix.shape[-1]
    if size == 1:
        return torch.exp(log_scale)[..., None, None] + matrix * 0.0
    rows = torch.arange(size, device=matrix.device).unsqueeze(1)
    cols = torch.arange(size - 1, device=matrix.device).unsqueeze(0)
    keep = torch.where(cols >= rows, cols + 1, cols)
    minors = matrix[..., keep[:, None, :, None], keep[None, :, None, :]]
    indices = torch.arange(size, device=matrix.device)
    parity = 1 - 2 * ((indices[:, None] + indices[None, :]) % 2)
    return parity * _cofactor_apply(minors, log_scale[..., None, None])


class _CofactorDeterminant(torch.autograd.Function):
    """Scaled determinant with an exact VJP, traceable by TorchDynamo."""

    @staticmethod
    def forward(ctx, matrix, log_scale):
        ctx.save_for_backward(matrix, log_scale)
        ctx.save_for_forward(matrix, log_scale)
        sign, log_abs = torch.linalg.slogdet(matrix)
        return sign * torch.exp(log_abs + log_scale)

    @staticmethod
    def backward(ctx, grad_output):
        matrix, log_scale = ctx.saved_tensors
        return grad_output[..., None, None] * _cofactors(matrix, log_scale), None


class _ForwardCofactorDeterminant(_CofactorDeterminant):
    """Same value/VJP, with forward AD support for eager kinetic evaluation."""

    @staticmethod
    def jvp(ctx, matrix_tangent, scale_tangent):
        matrix, log_scale = ctx.saved_tensors
        # The scale is a detached normalization, not a model variable.
        return (_cofactors(matrix, log_scale) * matrix_tangent).sum(dim=(-1, -2))


def _cofactor_apply(matrix, log_scale):
    # Dynamo currently rejects Function classes defining a custom jvp, even for
    # reverse-only programs. Both classes have identical values and VJPs.
    if torch._dynamo.is_compiling():
        return _CofactorDeterminant.apply(matrix, log_scale)
    return _ForwardCofactorDeterminant.apply(matrix, log_scale)


def _singular_mask(matrix, sign, log_abs):
    """Include determinants that are zero up to matrix-scale roundoff.

    LAPACK can return a tiny nonzero determinant for exactly singular matrices
    because elimination is finite precision.  Comparing the determinant with a
    norm-scaled machine-epsilon bound identifies those unresolved singular
    branches without treating uniformly scaled nonsingular matrices as zero.
    """
    size = matrix.shape[-1]
    magnitude = matrix.detach().abs().amax(dim=(-1, -2))
    positive = magnitude > 0
    safe_magnitude = torch.where(positive, magnitude, torch.ones_like(magnitude))
    log_bound = size * torch.log(safe_magnitude)
    log_bound = log_bound + math.log(size * torch.finfo(matrix.dtype).eps)
    return (sign == 0) | (positive & (log_abs <= log_bound))


def _determinant_terms(matrix):
    with torch.no_grad():
        raw_sign, raw_log_abs = torch.linalg.slogdet(matrix)
        singular = _singular_mask(matrix, raw_sign, raw_log_abs)
    identity = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
    safe = torch.where(singular[..., None, None], identity, matrix)
    sign, log_abs = torch.linalg.slogdet(safe)
    scale = log_abs.detach()
    amplitude = torch.where(
        singular, torch.zeros_like(sign), sign * torch.exp(log_abs - scale)
    )
    return amplitude, scale


def signed_determinant_sum(
    d_up: Optional[torch.Tensor],
    d_down: Optional[torch.Tensor],
    det_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (log|sum w det(up) det(down)|, sign), including zero branches."""
    matrix = d_up if d_up is not None else d_down
    if matrix is None:
        raise ValueError("At least one spin determinant is required")
    batch, count = matrix.shape[:2]
    one = matrix.new_ones(batch, count)
    zero = matrix.new_zeros(batch, count)
    a_up, s_up = _determinant_terms(d_up) if d_up is not None else (one, zero)
    a_down, s_down = _determinant_terms(d_down) if d_down is not None else (one, zero)
    amplitude = a_up * a_down
    scale = s_up + s_down
    singular = amplitude.detach() == 0
    maximum = scale.masked_fill(singular, float("-inf")).amax(dim=1, keepdim=True)
    maximum = torch.where(torch.isneginf(maximum), torch.zeros_like(maximum), maximum)
    relative = torch.where(singular, torch.zeros_like(scale), scale - maximum)
    scaled = amplitude * torch.exp(relative)

    if d_up is None:
        block = d_down
    elif d_down is None:
        block = d_up
    else:
        zeros = matrix.new_zeros(batch, count, d_up.shape[-1], d_down.shape[-1])
        block = torch.cat(
            (
                torch.cat((d_up, zeros), dim=-1),
                torch.cat((zeros.transpose(-1, -2), d_down), dim=-1),
            ),
            dim=-2,
        )
    row_scale = block.detach().abs().amax(dim=-1, keepdim=True)
    fallback = row_scale.amax(dim=1, keepdim=True)
    row_scale = torch.where(row_scale == 0, fallback, row_scale)
    row_scale = torch.where(row_scale == 0, torch.ones_like(row_scale), row_scale)
    normalized = block / row_scale
    correction_scale = row_scale.log().sum(dim=(-1, -2)) - maximum
    # Avoid unnecessary cofactor graphs on ordinary branches, without a data branch.
    identity = torch.eye(block.shape[-1], device=block.device, dtype=block.dtype)
    repair_input = torch.where(singular[..., None, None], normalized, identity)
    repair_scale = torch.where(
        singular, correction_scale, torch.zeros_like(correction_scale)
    )
    correction = _cofactor_apply(repair_input, repair_scale)
    scaled = torch.where(singular, correction - correction.detach(), scaled)
    summed = (det_weights.unsqueeze(0) * scaled).sum(dim=1)
    is_zero = summed == 0
    safe_abs = torch.where(is_zero, torch.ones_like(summed), summed.abs())
    log_abs = torch.where(
        is_zero,
        torch.full_like(summed, float("-inf")),
        maximum.squeeze(1) + safe_abs.log(),
    )
    return log_abs, summed.sign()
