"""Row-gathered SparseAdam for the per-splat parameters.

`torch.optim.SparseAdam` routes every step through sparse tensors: two
`sparse_mask` calls build a whole COO tensor each just to read the moment rows
the gradient touches, and three `make_sparse` constructions wrap the updates so
that `dense.add_(sparse)` can flatten their indices and land them with the same
`index_add_` this module calls directly. The per-splat gradients here always
carry `sparse_dim == 1` - one COO index per splat row, as
`SplatField._SparseRowGather` builds them - so the sparse wrapper adds kernels
and buys nothing.

The default CUDA fp32 path evaluates that same operation sequence in one Triton
kernel with an explicit round-to-nearest intrinsic for every arithmetic op. An
environment switch restores the original eager operations. Both paths are
bitwise identical to `torch.optim.SparseAdam`; `tests/test_row_sparse_adam.py`
asserts that over many steps, including across a refine event that resizes the
census. The state layout matches too - an integer `step` and dense `exp_avg`/
`exp_avg_sq` - so checkpoints written by either optimizer load into the other
and the refine surgery in `SplatRefinementStrategy._update_params` keeps
working unchanged.
"""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer

from survey_fusion.utils.perf_env import fused_row_sparse_adam_enabled


class RowSparseAdam(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        maximize: bool = False,
    ) -> None:
        if not 0.0 < lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 < eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "maximize": maximize}
        super().__init__(params, defaults)
        for group in self.param_groups:
            for param in group["params"]:
                if param.is_sparse:
                    raise ValueError(
                        "RowSparseAdam requires dense parameter tensors."
                    )
                if param.is_complex():
                    raise ValueError(
                        "RowSparseAdam does not support complex parameters."
                    )

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            maximize = group.get("maximize", False)
            for param in group["params"]:
                gradient = param.grad
                if gradient is None:
                    continue
                if not gradient.is_sparse:
                    raise RuntimeError(
                        "RowSparseAdam does not support dense gradients, please "
                        "consider Adam instead"
                    )
                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                # Counted before the empty-gradient skip below, as SparseAdam
                # counts it: the bias correction must not depend on which steps
                # happened to touch no row.
                state["step"] += 1
                step = state["step"]

                gradient = gradient if not maximize else -gradient
                # The update is non-linear, so a row must appear once.
                gradient = gradient.coalesce()
                assert gradient.is_coalesced()
                if gradient.sparse_dim() != 1:
                    raise ValueError(
                        "RowSparseAdam expects row-sparse gradients with one "
                        f"sparse dimension, got {gradient.sparse_dim()}."
                    )
                values = gradient._values()
                if values.numel() == 0:
                    continue
                rows = gradient._indices()[0]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                use_fused = False
                if fused_row_sparse_adam_enabled() and param.is_cuda:
                    from . import row_sparse_adam_kernels

                    use_fused = row_sparse_adam_kernels.applies(
                        param, exp_avg, exp_avg_sq, values, rows
                    )
                if use_fused:
                    row_sparse_adam_kernels.update_rows_(
                        param,
                        exp_avg,
                        exp_avg_sq,
                        values,
                        rows,
                        one_minus_beta1=1 - beta1,
                        one_minus_beta2=1 - beta2,
                        eps=eps,
                        neg_step_size=-step_size,
                    )
                else:
                    # `exp_avg.sparse_mask(gradient)` copies exactly these rows.
                    old_exp_avg = exp_avg.index_select(0, rows)
                    exp_avg_update = values.sub(old_exp_avg).mul_(1 - beta1)
                    exp_avg.index_add_(0, rows, exp_avg_update)
                    old_exp_avg_sq = exp_avg_sq.index_select(0, rows)
                    exp_avg_sq_update = (
                        values.pow(2).sub_(old_exp_avg_sq).mul_(1 - beta2)
                    )
                    exp_avg_sq.index_add_(0, rows, exp_avg_sq_update)

                    # Adding back the old values is cheaper than a second gather.
                    numer = exp_avg_update.add_(old_exp_avg)
                    exp_avg_sq_update.add_(old_exp_avg_sq)
                    denom = exp_avg_sq_update.sqrt_().add_(eps)
                    param.index_add_(0, rows, -step_size * numer.div_(denom))
        return loss
