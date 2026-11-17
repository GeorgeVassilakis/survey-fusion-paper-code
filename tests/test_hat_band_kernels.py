"""The fused band weights must reproduce the eager chain bit for bit.

Every comparison is on the raw bit patterns, not on values: a weight of -0.0
where the reference holds +0.0 would pass `torch.equal` and still mean the
fused path reassociated something.
"""

from __future__ import annotations

import math

import pytest
import torch

from survey_fusion.models.splat_field import SplatField
from survey_fusion.pgm.splat_spectrum import (
    _hat_band_tables,
    _hat_fused_backend,
    _hat_interior_bounds,
    _hat_weights_and_grad,
    _hat_window_width,
    splat_band_amplitudes,
)
from survey_fusion.utils.perf_env import fused_band_weights_enabled

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the fused band weights need CUDA"
)

LOGLAM_MIN, LOGLAM_MAX = math.log(2980.0), math.log(11230.0)
SCALE = 0.5 * (LOGLAM_MAX - LOGLAM_MIN)


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.numel() == 0:
        return True
    return torch.equal(
        a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)
    )


def _responses() -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Bandpasses spanning coarse, fine and grid-wide segment tables."""
    out = {}
    lam = torch.arange(5380.0, 7231.0, 25.0, dtype=torch.float64)
    edges = torch.minimum((lam - 5400.0) / 150.0, (7200.0 - lam) / 150.0)
    out["plateau"] = (lam, 0.55 * edges.clamp(0.0, 1.0))
    for key, (lo, hi, nodes) in {
        "coarse": (2980.0, 4130.0, 12),
        "fine": (7730.0, 11230.0, 141),
        "wide": (3000.0, 11200.0, 40),
    }.items():
        lam = torch.linspace(lo, hi, nodes, dtype=torch.float64)
        arc = torch.linspace(0.0, math.pi, nodes, dtype=torch.float64)
        out[key] = (lam, torch.sin(arc))
    return out


def _grid(G: int, device) -> torch.Tensor:
    grid = torch.linspace(-1.0, 1.0, G, dtype=torch.float32, device=device)
    return torch.exp(0.5 * (LOGLAM_MAX + LOGLAM_MIN) + SCALE * grid)


def _shifts(kind: str, rows: int, device) -> torch.Tensor:
    generator = torch.Generator().manual_seed(rows + len(kind))
    if kind == "narrow":
        shift = torch.randn(rows, generator=generator) * 0.01
    else:
        shift = torch.rand(rows, generator=generator) * 6.0 - 3.0
    return torch.exp(SCALE * shift).to(device=device, dtype=torch.float32)


def _eager(monkeypatch, *args, **kwargs):
    monkeypatch.setenv("SPLAT_AMP_TRITON", "0")
    try:
        return _hat_weights_and_grad(*args, **kwargs)
    finally:
        monkeypatch.setenv("SPLAT_AMP_TRITON", "1")


@requires_cuda
def test_fused_weights_match_eager_bit_for_bit(monkeypatch) -> None:
    device = torch.device("cuda")
    checked = 0
    for name, (lam, thr) in _responses().items():
        tables = _hat_band_tables(
            lam.to(device), thr.to(device), dtype=torch.float32, device=device
        )
        for G in (96, 32):
            C = _grid(G, device)
            width = _hat_window_width(C, tables[-2], tables[-1])
            for window in dict.fromkeys((width, G)):
                for kind in ("narrow", "spread"):
                    for rows in (0, 1, 2, 3, 7, 65, 4096):
                        t = _shifts(kind, rows, device)
                        for drop in (True, False):
                            for need_grad in (True, False):
                                kwargs = {
                                    "need_grad": need_grad, "window": window,
                                    "drop_off_grid": drop,
                                }
                                got = _hat_weights_and_grad(t, C, *tables, **kwargs)
                                want = _eager(monkeypatch, t, C, *tables, **kwargs)
                                label = (
                                    f"{name} G={G} window={window} {kind} "
                                    f"rows={rows} drop={drop} grad={need_grad}"
                                )
                                assert _same_bits(got[0], want[0]), f"W: {label}"
                                if need_grad:
                                    assert _same_bits(got[1], want[1]), f"dW: {label}"
                                else:
                                    assert got[1] is None and want[1] is None
                                checked += 1
    assert checked > 500


@requires_cuda
def test_fused_band_amplitudes_and_gradients_match(monkeypatch) -> None:
    device = torch.device("cuda")
    responses = _responses()
    bandpasses = {
        key: (lam.to(torch.float32), thr.to(torch.float32))
        for key, (lam, thr) in responses.items()
        if key != "wide"
    }
    for off_grid in ("zero", "clamp"):
        for subsets in (True, False):
            outputs = []
            for enabled in ("1", "0"):
                monkeypatch.setenv("SPLAT_AMP_TRITON", enabled)
                torch.manual_seed(5)
                field = SplatField(
                    n_splats=2048, n_templates=16,
                    loglam_grid=torch.linspace(-1.0, 1.0, 96),
                    init_scale=0.002, sed_off_grid=off_grid,
                ).to(device)
                with torch.no_grad():
                    field.sed_shift.normal_(0.0, 0.05)
                field.set_sparse_gradients(True)
                indices = torch.arange(0, 2048, 2, device=device)
                rows = indices.numel()
                band_rows = None
                if subsets:
                    band_rows = [
                        torch.arange(i, rows, 3 + i, device=device)
                        for i in range(len(bandpasses))
                    ]
                amplitudes = splat_band_amplitudes(
                    field, bandpasses,
                    loglam_min=LOGLAM_MIN, loglam_max=LOGLAM_MAX,
                    indices=indices, indices_sorted_unique=True,
                    band_row_indices=band_rows,
                )
                stacked = torch.stack(
                    [amplitudes[key] for key in bandpasses], dim=1
                )
                torch.manual_seed(6)
                (stacked * torch.randn_like(stacked)).sum().backward()
                captured = {"amplitudes": stacked.detach()}
                for name in ("sed_coeffs", "raw_flux", "sed_shift"):
                    grad = getattr(field, name).grad.coalesce()
                    captured[f"{name}.indices"] = grad.indices()
                    captured[f"{name}.values"] = grad.values()
                captured["templates"] = field.sed.raw_templates.grad
                outputs.append(captured)
            monkeypatch.setenv("SPLAT_AMP_TRITON", "1")
            fused, eager = outputs
            for key in eager:
                label = f"{key} off_grid={off_grid} subsets={subsets}"
                if key.endswith(".indices"):
                    assert torch.equal(fused[key], eager[key]), label
                else:
                    assert _same_bits(fused[key], eager[key]), label


@requires_cuda
def test_float32_segment_bounds_decide_the_same_search() -> None:
    from survey_fusion.pgm import hat_band_kernels

    device = torch.device("cuda")
    for lam, thr in _responses().values():
        tables = _hat_band_tables(
            lam.to(device), thr.to(device), dtype=torch.float32, device=device
        )
        interior = _hat_interior_bounds(tables[0])
        bounds = hat_band_kernels.float32_segment_bounds(interior)
        assert bounds.dtype is torch.float32
        assert torch.equal(bounds, bounds.sort().values)
        # Queries land on, just under and just over every boundary.
        exact = interior.to(torch.float32)
        query = torch.cat([
            exact,
            torch.nextafter(exact, torch.full_like(exact, -math.inf)),
            torch.nextafter(exact, torch.full_like(exact, math.inf)),
            torch.linspace(
                tables[-2], tables[-1], 4096, dtype=torch.float32, device=device
            ),
        ])
        want = torch.searchsorted(
            interior, query.to(torch.float64).contiguous(), right=True
        )
        got = torch.searchsorted(bounds, query.contiguous(), right=True)
        assert torch.equal(got, want)


@requires_cuda
def test_fused_backend_is_selected_on_cuda(monkeypatch) -> None:
    t = torch.zeros(8, dtype=torch.float32, device="cuda")
    assert _hat_fused_backend(t, 25, 96) is not None
    assert _hat_fused_backend(t.double(), 25, 96) is None
    assert _hat_fused_backend(t, 1, 96) is None
    monkeypatch.setenv("SPLAT_AMP_TRITON", "0")
    assert _hat_fused_backend(t, 25, 96) is None


def test_fused_backend_declines_without_cuda() -> None:
    t = torch.zeros(8, dtype=torch.float32)
    assert _hat_fused_backend(t, 25, 96) is None


def test_fused_backend_is_enabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("SPLAT_AMP_TRITON", raising=False)
    assert fused_band_weights_enabled()
    monkeypatch.setenv("SPLAT_AMP_TRITON", "0")
    assert not fused_band_weights_enabled()
