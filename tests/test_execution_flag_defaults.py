"""The paper runs' execution path is the default; each switch still restores the reference."""
import pytest

from survey_fusion.utils import perf_env

PAPER_SWITCHES = {
    "SPLAT_NO_CHECKPOINT": perf_env.activation_checkpointing_disabled,
    "SPLAT_EIGEN_FAST_CONTEXT": perf_env.eigen_fast_context_enabled,
    "SPLAT_EIGEN_SHAPE_CACHE": perf_env.eigen_shape_cache_enabled,
    "SPLAT_EIGEN_FUSED_INDEX": perf_env.eigen_fused_index_enabled,
    "SPLAT_EIGEN_REUSE_GATHER": perf_env.eigen_reuse_gather_enabled,
    "SPLAT_EIGEN_FUSED_PREDICT": perf_env.eigen_fused_predict_enabled,
    "SPLAT_EIGEN_STRICT_VJP": perf_env.eigen_strict_vjp_enabled,
    "SPLAT_EIGEN_FULL_FMA": perf_env.eigen_full_fma_enabled,
    "SPLAT_EIGEN_ASYNC_VALIDATE": perf_env.eigen_async_validate_enabled,
    "SPLAT_COALESCE_DIRECT": perf_env.coalesce_direct_enabled,
    "SPLAT_COMPACT_ALL_BANDS": perf_env.compact_all_bands_enabled,
    "SPLAT_PRUNE_ONLY_FAST": perf_env.prune_only_fast_enabled,
    "SPLAT_FAST_SPECTRAL_GROUPS": perf_env.fast_spectral_groups_enabled,
}


@pytest.mark.parametrize("name", sorted(PAPER_SWITCHES))
def test_paper_switch_is_on_by_default_and_zero_restores_the_reference(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(name, raising=False)
    assert PAPER_SWITCHES[name]() is True
    monkeypatch.setenv(name, "0")
    assert PAPER_SWITCHES[name]() is False


@pytest.mark.parametrize("name", sorted(PAPER_SWITCHES))
def test_paper_switch_refuses_other_spellings(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(name, "true")
    with pytest.raises(ValueError, match=name):
        PAPER_SWITCHES[name]()


def test_execution_flags_record_resolved_switches_and_raw_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPLAT_EIGEN_FULL_FMA", "0")
    monkeypatch.setenv("SPLAT_AMP_CHUNK", "8000000")
    flags = perf_env.execution_flags()
    assert flags["resolved"]["SPLAT_EIGEN_FULL_FMA"] is False
    assert flags["resolved"]["SPLAT_PRUNE_ONLY_FAST"] is True
    assert set(PAPER_SWITCHES) <= set(flags["resolved"])
    assert flags["environment"]["SPLAT_AMP_CHUNK"] == "8000000"
    assert all(name.startswith("SPLAT_") for name in flags["environment"])
