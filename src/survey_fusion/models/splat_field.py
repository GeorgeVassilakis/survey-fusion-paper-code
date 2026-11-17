"""Gaussian-splat spectral field.

F(x, y, loglam) = sum_j w_j G_j(x, y) S_j(loglam)

Each splat j is a 2D elliptical Gaussian on the tangent plane with a spectrum
drawn from a shared learned SED dictionary. This is the source term only: the
sky is not a model parameter. The observation model supplies the SDSS-measured
sky as a known additive offset (see the trainer's observation contract), and the
data likelihood is over source + sky. A second, free sky term here would be both
redundant and, being strictly positive, unable to correct the known sky downward.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from survey_fusion.models.sed_dictionary import SEDDictionary, _rng_reference_count
from survey_fusion.utils.perf_env import fused_covariance_enabled


class _SparseRowGather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, parameter: torch.Tensor, indices: torch.Tensor,
        sorted_unique: bool = False,
    ) -> torch.Tensor:
        indices = indices.to(device=parameter.device, dtype=torch.long)
        ctx.save_for_backward(indices)
        ctx.parameter_shape = parameter.shape
        ctx.sorted_unique = bool(sorted_unique)
        return torch.index_select(parameter, 0, indices)

    @staticmethod
    def backward(ctx, output_gradient: torch.Tensor):
        (indices,) = ctx.saved_tensors
        gradient = torch.sparse_coo_tensor(
            indices.reshape(1, -1),
            output_gradient,
            size=ctx.parameter_shape,
            device=output_gradient.device,
            dtype=output_gradient.dtype,
            check_invariants=False,
            # Callers assert sorted_unique only for index sets built by
            # torch.unique(sorted=True) or nonzero, whose ascending-unique
            # order makes the COO coalesced by construction; claiming it
            # skips a full sort per gather. With duplicate indices this
            # claim would silently corrupt the gradient, so it must never
            # be passed for arbitrary index sets.
            is_coalesced=ctx.sorted_unique,
        )
        if not ctx.sorted_unique:
            gradient = gradient.coalesce()
        return gradient, None, None


class _FusedSparseRowGather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        indices: torch.Tensor,
        sorted_unique: bool,
        *parameters: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        indices = indices.to(device=parameters[0].device, dtype=torch.long)
        ctx.save_for_backward(indices)
        ctx.parameter_shapes = tuple(parameter.shape for parameter in parameters)
        ctx.sorted_unique = bool(sorted_unique)
        # Separate gathers leave an unused parameter's gradient absent. A zero
        # gradient would instead advance Adam and apply its existing momentum.
        ctx.set_materialize_grads(False)
        from survey_fusion.pgm.splat_gather_kernels import gather_splat_rows

        return gather_splat_rows(indices, *parameters)

    @staticmethod
    def backward(ctx, *output_gradients: torch.Tensor):
        (indices,) = ctx.saved_tensors
        gradients = []
        for output_gradient, shape in zip(output_gradients, ctx.parameter_shapes):
            if output_gradient is None:
                gradients.append(None)
                continue
            gradient = torch.sparse_coo_tensor(
                indices.reshape(1, -1),
                output_gradient,
                size=shape,
                device=output_gradient.device,
                dtype=output_gradient.dtype,
                check_invariants=False,
                is_coalesced=ctx.sorted_unique,
            )
            if not ctx.sorted_unique:
                gradient = gradient.coalesce()
            gradients.append(gradient)
        return (None, None, *gradients)


class SplatField(nn.Module):
    def __init__(
        self,
        n_splats: int,
        n_templates: int,
        loglam_grid: torch.Tensor,
        init_extent: float = 1.0,
        init_scale: float = 0.02,
        sed_off_grid: str = "clamp",
        sed_pin_gauge_knots: bool = False,
        sed_pad_knots_blue: int = 0,
        sed_pad_knots_red: int = 0,
        sed_rng_reference_templates: int | None = None,
    ):
        super().__init__()
        self.mu = nn.Parameter((torch.rand(n_splats, 2) * 2 - 1) * init_extent)
        self.log_scales = nn.Parameter(
            torch.full((n_splats, 2), math.log(init_scale))
        )
        self.theta = nn.Parameter(torch.zeros(n_splats))
        self.raw_flux = nn.Parameter(torch.full((n_splats,), -4.0))
        self.sed = SEDDictionary(
            n_templates,
            loglam_grid,
            off_grid=sed_off_grid,
            pin_gauge_knots=sed_pin_gauge_knots,
            pad_knots_blue=sed_pad_knots_blue,
            pad_knots_red=sed_pad_knots_red,
            rng_reference_templates=sed_rng_reference_templates,
        )
        # Coefficients must also start unequal: with identical coefficients
        # every template still sees the same gradient. 0.01*randn is far too
        # small to break that; use unit scale so splats begin with genuinely
        # different spectral mixes.
        rng_templates = _rng_reference_count(n_templates, sed_rng_reference_templates)
        self.sed_coeffs = nn.Parameter(
            torch.randn(n_splats, rng_templates)[:, :n_templates].contiguous()
        )
        # per-splat translation in normalized natural-log-wavelength units;
        # redshift z enters as ln(1+z) = scale * shift, learned freely per splat
        self.sed_shift = nn.Parameter(torch.zeros(n_splats))
        # Lineage record, maintained by refine surgery: how many times a row's
        # flux has been halved by reproduction, the step of the last halving,
        # and the step a row was born at (-1 = present since init). Persistent
        # buffers so checkpoints carry the census history.
        self.register_buffer(
            "halving_depth", torch.zeros(n_splats, dtype=torch.uint8)
        )
        self.register_buffer(
            "last_halving_event", torch.full((n_splats,), -1, dtype=torch.int32)
        )
        self.register_buffer(
            "birth_event", torch.full((n_splats,), -1, dtype=torch.int32)
        )
        self._sparse_gradients = False
        self._sparse_gather_log: dict[int, list[torch.Tensor] | None] | None = None

    # --- parameter views -------------------------------------------------
    def set_sparse_gradients(self, enabled: bool) -> None:
        self._sparse_gradients = bool(enabled)

    @property
    def sparse_gradients_enabled(self) -> bool:
        return self._sparse_gradients

    def start_sparse_gather_log(self) -> None:
        """Begin recording which index tensors each parameter is gathered with.

        A sparse gradient is the concatenation of one COO block per gather, so
        the rows it holds are exactly the union of the recorded index sets.
        Only gathers whose caller guarantees sorted, unique indices qualify
        for sort-free assembly. None marks a parameter with an unverified
        gather, which must keep ordinary coalescence for the entire step.
        """
        self._sparse_gather_log = {}

    def take_sparse_gather_log(self) -> dict[int, list[torch.Tensor] | None] | None:
        """Return the recorded gathers and stop recording."""
        log = self._sparse_gather_log
        self._sparse_gather_log = None
        return log

    def _gather_parameter(
        self, parameter: torch.Tensor, indices: torch.Tensor | None,
        *, sorted_unique: bool = False,
    ) -> torch.Tensor:
        if indices is None:
            return parameter
        indices = indices.to(device=parameter.device, dtype=torch.long)
        if self._sparse_gradients and torch.is_grad_enabled():
            if self._sparse_gather_log is not None:
                key = id(parameter)
                if not sorted_unique:
                    self._sparse_gather_log[key] = None
                else:
                    gathered = self._sparse_gather_log.setdefault(key, [])
                    if gathered is not None:
                        gathered.append(indices)
            return _SparseRowGather.apply(parameter, indices, sorted_unique)
        return torch.index_select(parameter, 0, indices)

    def gather_splat_parameters(
        self, indices: torch.Tensor, *, sorted_unique: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Gather all six per-splat parameter arrays in one CUDA launch."""
        indices = indices.to(device=self.mu.device, dtype=torch.long)
        parameters = self.per_splat_parameters()
        use_fused = self._sparse_gradients and self.mu.is_cuda
        if use_fused:
            from survey_fusion.pgm.splat_gather_kernels import applies, gather_splat_rows

            use_fused = applies(indices, *parameters)
        if not use_fused:
            return tuple(
                self._gather_parameter(
                    parameter, indices, sorted_unique=sorted_unique
                )
                for parameter in parameters
            )
        if (
            torch.is_grad_enabled()
            and self._sparse_gather_log is not None
        ):
            for parameter in parameters:
                key = id(parameter)
                if not sorted_unique:
                    self._sparse_gather_log[key] = None
                else:
                    gathered = self._sparse_gather_log.setdefault(key, [])
                    if gathered is not None:
                        gathered.append(indices)
        if torch.is_grad_enabled():
            return _FusedSparseRowGather.apply(
                indices, bool(sorted_unique), *parameters
            )
        return gather_splat_rows(indices, *parameters)

    def positions(
        self, indices: torch.Tensor | None = None,
        *, sorted_unique: bool = False,
    ) -> torch.Tensor:
        return self._gather_parameter(self.mu, indices, sorted_unique=sorted_unique)

    def covariances(
        self, indices: torch.Tensor | None = None,
        *, sorted_unique: bool = False,
    ) -> torch.Tensor:
        log_scales = self._gather_parameter(
            self.log_scales, indices, sorted_unique=sorted_unique
        )
        theta = self._gather_parameter(
            self.theta, indices, sorted_unique=sorted_unique
        )
        s = torch.exp(log_scales)
        c, snt = torch.cos(theta), torch.sin(theta)
        R = torch.stack(
            [torch.stack([c, -snt], -1), torch.stack([snt, c], -1)], -2
        )
        D = torch.diag_embed(s**2)
        return R @ D @ R.transpose(-1, -2)

    def shape_rows(
        self, indices: torch.Tensor | None = None,
        *, sorted_unique: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(log_scales, theta) rows: the raw shape parameters of each splat.

        The rows carry the small axis exactly, which the fp32 covariance
        entries built from them do not: a needle's minor eigenvalue is a
        cancellation of a*d against b*b there, so callers that need a
        determinant or an inverse of a badly conditioned splat take it from
        these rows instead.
        """
        log_scales = self._gather_parameter(
            self.log_scales, indices, sorted_unique=sorted_unique
        )
        theta = self._gather_parameter(
            self.theta, indices, sorted_unique=sorted_unique
        )
        return log_scales, theta

    def covariance_components(
        self, indices: torch.Tensor | None = None,
        *, sorted_unique: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(a, b, d) of the symmetric 2x2 covariance, computed elementwise.

        Same matrix as `covariances` without materializing R or running the
        batched 2x2 matmuls; b is returned once since the matrix is symmetric
        by construction.
        """
        log_scales = self._gather_parameter(
            self.log_scales, indices, sorted_unique=sorted_unique
        )
        theta = self._gather_parameter(
            self.theta, indices, sorted_unique=sorted_unique
        )
        return self.covariance_components_from_rows(log_scales, theta)

    @staticmethod
    def covariance_components_from_rows(
        log_scales: torch.Tensor, theta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if fused_covariance_enabled() and log_scales.is_cuda:
            from survey_fusion.pgm import covariance_kernels

            if covariance_kernels.applies(log_scales, theta):
                return covariance_kernels.Covariance.apply(log_scales, theta)
        return SplatField._covariance_components_eager_from_rows(log_scales, theta)

    @staticmethod
    def _covariance_components_eager_from_rows(
        log_scales: torch.Tensor, theta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = torch.exp(2.0 * log_scales[:, 0])
        e2 = torch.exp(2.0 * log_scales[:, 1])
        c, s = torch.cos(theta), torch.sin(theta)
        a = c * c * e1 + s * s * e2
        b = c * s * (e1 - e2)
        d = s * s * e1 + c * c * e2
        return a, b, d

    def fluxes(
        self, indices: torch.Tensor | None = None,
        *, sorted_unique: bool = False,
    ) -> torch.Tensor:
        return F.softplus(
            self._gather_parameter(
                self.raw_flux, indices, sorted_unique=sorted_unique
            )
        )

    def raw_flux_values(
        self, indices: torch.Tensor | None = None,
        *, sorted_unique: bool = False,
    ) -> torch.Tensor:
        return self._gather_parameter(
            self.raw_flux, indices, sorted_unique=sorted_unique
        )

    def sed_coefficients(
        self, indices: torch.Tensor | None = None,
        *, sorted_unique: bool = False,
    ) -> torch.Tensor:
        return self._gather_parameter(
            self.sed_coeffs, indices, sorted_unique=sorted_unique
        )

    def spectral_shifts(
        self, indices: torch.Tensor | None = None,
        *, sorted_unique: bool = False,
    ) -> torch.Tensor:
        return self._gather_parameter(
            self.sed_shift, indices, sorted_unique=sorted_unique
        )

    def per_splat_parameters(self) -> tuple[nn.Parameter, ...]:
        return (
            self.mu,
            self.log_scales,
            self.theta,
            self.raw_flux,
            self.sed_coeffs,
            self.sed_shift,
        )

    def sparse_param_groups_gsplat(self, scene_scale: float = 1.0) -> list[dict]:
        """gsplat default learning rates for the per-splat parameters, for
        SparseAdam under sparse_splat_gradients. Same mapping as
        param_groups_gsplat; the shared SED dictionary lives in
        shared_param_groups_gsplat."""
        return [
            {"params": [self.mu], "lr": 1.6e-4 * scene_scale, "name": "mu"},
            {"params": [self.log_scales], "lr": 5e-3, "name": "log_scales"},
            {"params": [self.theta], "lr": 1e-3, "name": "theta"},
            {"params": [self.raw_flux], "lr": 5e-2, "name": "raw_flux"},
            {"params": [self.sed_coeffs], "lr": 2.5e-3, "name": "sed_coeffs"},
            {"params": [self.sed_shift], "lr": 2.5e-3 / 20, "name": "sed_shift"},
        ]

    def shared_param_groups_gsplat(self) -> list[dict]:
        return [{"params": list(self.sed.parameters()), "lr": 2.5e-3, "name": "sed"}]

    def param_groups_gsplat(self, scene_scale: float = 1.0) -> list[dict]:
        """gsplat/3DGS default learning rates (simple_trainer.py defaults),
        mapped: means->mu, scales->log_scales, quats->theta,
        opacities->raw_flux, sh0->sed_coeffs/templates/sky, shN->sed_shift.
        Pair mu with ExponentialLR decay over the full schedule as gsplat does."""
        return [
            {"params": [self.mu], "lr": 1.6e-4 * scene_scale, "name": "mu"},
            {"params": [self.log_scales], "lr": 5e-3, "name": "log_scales"},
            {"params": [self.theta], "lr": 1e-3, "name": "theta"},
            {"params": [self.raw_flux], "lr": 5e-2, "name": "raw_flux"},
            {"params": [self.sed_coeffs, *self.sed.parameters()], "lr": 2.5e-3, "name": "sed"},
            {"params": [self.sed_shift], "lr": 2.5e-3 / 20, "name": "sed_shift"},
        ]

    # --- evaluation ------------------------------------------------------
    def forward(
        self, x: torch.Tensor, y: torch.Tensor, log_lam: torch.Tensor,
        max_elements: int = 200_000_000,
    ) -> torch.Tensor:
        """Evaluate the field at n query points.

        The Gaussian intermediates are (n_query, n_splats), so a caller asking
        for many points at once allocates n*J floats several times over: 12k
        splats against a 640k-point Monte-Carlo batch is 7.7e9 elements, ~94 GiB.
        Chunk over query points so the peak stays near max_elements regardless
        of what the caller asks for. Under grad, each chunk is checkpointed so
        the saved activations do not simply reappear in the backward graph.

        SEDDictionary.eval bounds its own (n_templates, n_splats, n_query)
        intermediate, so this budget is on n*J, the shape actually returned.
        Scaling this chunk down by n_templates instead would make chunks 32x
        smaller and 32x more numerous, each separately checkpointed, which is
        correct but far too slow to finish a step.
        """
        n = int(x.shape[0])
        j = int(self.mu.shape[0])
        step = max(1, int(max_elements // max(j, 1)))
        if n <= step:
            return self._forward_chunk(x, y, log_lam)
        outs = []
        for start in range(0, n, step):
            sl = slice(start, min(start + step, n))
            if torch.is_grad_enabled():
                outs.append(
                    checkpoint(self._forward_chunk, x[sl], y[sl], log_lam[sl],
                               use_reentrant=False)
                )
            else:
                outs.append(self._forward_chunk(x[sl], y[sl], log_lam[sl]))
        return torch.cat(outs, dim=0)

    def _forward_chunk(
        self, x: torch.Tensor, y: torch.Tensor, log_lam: torch.Tensor
    ) -> torch.Tensor:
        d = torch.stack([x, y], -1)[:, None, :] - self.mu[None, :, :]
        cov = self.covariances()
        prec = torch.linalg.inv(cov)
        q = torch.einsum("njk,jkl,njl->nj", d, prec, d)
        norm = 1.0 / (2 * math.pi * torch.sqrt(torch.linalg.det(cov)))
        G = norm[None, :] * torch.exp(-0.5 * q)
        S = self.sed.eval(self.sed_coeffs, log_lam, self.sed_shift)  # (J, n)
        return (G * (self.fluxes()[None, :] * S.T)).sum(-1)
