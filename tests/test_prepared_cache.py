import numpy as np
import json
import os
import pickle
from types import SimpleNamespace
import pytest
import torch
from omegaconf import OmegaConf

from survey_fusion.data.prepared import load_prepared, prepared_digest, save_prepared

from test_analytic_trainer_dispatch import _bandpasses, _realistic_scene


def _build_store():
    from survey_fusion.data.adapters.sdss import build_sdss_datums

    return build_sdss_datums(
        _realistic_scene(), _bandpasses(),
        psf_fwhm_arcsec=1.0, device=torch.device("cpu"),
        spectral_response_mode="legacy_imaging_median",
    )


def test_prepared_roundtrip_preserves_datums(tmp_path):
    scene = _realistic_scene()
    store = _build_store()
    path = tmp_path / "prepared_test.pt"
    save_prepared(path, scene, store)
    loaded = load_prepared(path)
    assert loaded is not None
    scene2, store2 = loaded

    idx = torch.arange(min(64, store.batch.y.numel()))
    for col in ("y", "sigma", "exposure_id", "datum_kind", "datum_index"):
        a = store.gather_rows(col, idx)
        b = store2.gather_rows(col, idx)
        assert torch.equal(a, b), f"column {col} changed across save/load"
    assert store2.n_image == store.n_image
    assert store2.n_spec == store.n_spec
    assert store2.image_band_by_exposure == store.image_band_by_exposure
    for band, img in scene.images_flambda.items():
        np.testing.assert_array_equal(np.asarray(scene2.images_flambda[band]), img)


def test_prepared_load_restores_digest_version_from_bound_manifest(tmp_path):
    digest = "d" * 64
    manifest = tmp_path / "scene.json"
    manifest.write_text(
        json.dumps({
            "psf_mixture_digest": digest,
            "psf_mixture_digest_version": 2,
        }),
        encoding="utf-8",
    )
    scene = SimpleNamespace(metadata={"psf_mixture_digest": digest})
    store = SimpleNamespace(
        psf_mixture_digest=digest,
        psf_mixture_digest_version=None,
    )
    path = tmp_path / "prepared_psf"
    kwargs = {
        "data_cfg": {}, "seed": 42, "bands": ["r"],
        "manifest_path": manifest,
    }
    save_prepared(path, scene, store, **kwargs)

    _loaded_scene, loaded_store = load_prepared(path, **kwargs)

    assert loaded_store.psf_mixture_digest_version == 2


def test_prepared_load_missing_or_corrupt_returns_none(tmp_path):
    assert load_prepared(tmp_path / "absent.pt") is None
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a torch archive")
    assert load_prepared(bad) is None


def test_prepared_old_schema_is_refused_for_clean_rebuild(tmp_path, capsys):
    from survey_fusion.data.prepared import PREPARED_SCHEMA

    path = tmp_path / "old_schema.pt"
    save_prepared(path, "scene", "store")
    meta = path / "meta.pkl"
    payload = pickle.loads(meta.read_bytes())
    payload["schema"] = PREPARED_SCHEMA - 1
    meta.write_bytes(pickle.dumps(payload))
    assert load_prepared(path) is None
    assert "rebuilding" in capsys.readouterr().out


def test_prepared_load_raises_on_transient_resource_failure(tmp_path, monkeypatch):
    # fd exhaustion while mmapping sidecars is the caller's environment, not
    # cache damage; the rebuild path deletes a healthy cache in place, so
    # load_prepared must raise instead of falling back
    import errno

    import survey_fusion.data.prepared as prepared_mod

    path = tmp_path / "prepared_ok.pt"
    save_prepared(path, 1, 2)

    for transient in (errno.EMFILE, errno.ENFILE, errno.ENOMEM,
                      errno.EAGAIN, errno.ESTALE, errno.ETIMEDOUT):
        def _exhausted(self, _code=transient):
            raise OSError(_code, "transient resource failure")

        monkeypatch.setattr(prepared_mod._SidecarUnpickler, "load", _exhausted)
        with pytest.raises(RuntimeError, match="not at fault"):
            load_prepared(path)

    def _missing_sidecar(self):
        raise FileNotFoundError(errno.ENOENT, "sidecar gone")

    monkeypatch.setattr(prepared_mod._SidecarUnpickler, "load", _missing_sidecar)
    assert load_prepared(path) is None


def test_prepared_digest_tracks_inputs(tmp_path):
    base = dict(observation_root="/x", scene_manifest_path="m.json")
    d0 = prepared_digest(base, seed=0, bands=["g", "r"], manifest_path=None)
    assert d0 == prepared_digest(
        base, seed=0, bands=["r", "g"], manifest_path=None
    ), "band order must not matter"
    assert d0 != prepared_digest(base, seed=1, bands=["g", "r"], manifest_path=None)
    changed = dict(base, spec_min_ivar=0.5)
    assert d0 != prepared_digest(changed, seed=0, bands=["g", "r"], manifest_path=None)
    ignored = dict(base, loader_workers=4, datum_spill_dir="/tmp/s")
    assert d0 == prepared_digest(ignored, seed=0, bands=["g", "r"], manifest_path=None)


def test_prepared_digest_tracks_manifest_content_not_mtime(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"value":"aaaa"}', encoding="utf-8")
    original_stat = manifest.stat()
    kwargs = {
        "seed": 3, "bands": ["r"], "manifest_path": manifest,
    }
    before = prepared_digest({}, **kwargs)

    manifest.write_text('{"value":"bbbb"}', encoding="utf-8")
    os.utime(manifest, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    after = prepared_digest({}, **kwargs)

    assert before != after
    assert len(before) == 64


def test_prepared_digest_ignores_manifest_mtime_only_touch(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"value":1}', encoding="utf-8")
    kwargs = {
        "seed": 3, "bands": ["r"], "manifest_path": manifest,
    }
    before = prepared_digest({}, **kwargs)
    stat = manifest.stat()
    os.utime(manifest, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert prepared_digest({}, **kwargs) == before


def test_prepared_load_rejects_stored_digest_mismatch(tmp_path):
    path = tmp_path / "prepared_mismatch"
    save_prepared(path, "scene", "store")
    meta = path / "meta.pkl"
    payload = pickle.loads(meta.read_bytes())
    payload["prepared_digest"] = "0" * 64
    meta.write_bytes(pickle.dumps(payload))

    with pytest.raises(RuntimeError, match="Prepared cache digest mismatch"):
        load_prepared(path)


def test_prepared_save_rejects_inputs_changed_after_cache_keying(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"value":1}', encoding="utf-8")
    expected = prepared_digest(
        {}, seed=0, bands=["r"], manifest_path=manifest,
    )
    manifest.write_text('{"value":2}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="inputs changed while the scene was being built"):
        save_prepared(
            tmp_path / "prepared_stale", "scene", "store", data_cfg={}, seed=0,
            bands=["r"], manifest_path=manifest, expected_digest=expected,
        )


def test_distributed_prepare_requires_a_readable_cache(tmp_path, monkeypatch):
    import train

    monkeypatch.setattr(train, "load_prepared", lambda _path, **_kwargs: None)
    monkeypatch.setattr(
        train,
        "load_sdss_scene",
        lambda *_args, **_kwargs: pytest.fail("distributed rank attempted a cache rebuild"),
    )

    with pytest.raises(RuntimeError, match="requires a readable prepared cache"):
        train.prepare_scene_and_datums(
            {"prepared_dir": str(tmp_path)},
            _bandpasses(),
            trainer_cfg={},
            seed=42,
            device=torch.device("cpu"),
            require_prepared_cache=True,
        )


def test_build_components_can_require_prepared_cache(monkeypatch, tmp_path):
    import train

    class ExpectedStop(RuntimeError):
        pass

    def fake_prepare(*_args, require_prepared_cache=False, **_kwargs):
        assert require_prepared_cache is True
        raise ExpectedStop

    monkeypatch.setattr(train, "load_bandpasses", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(train, "prepare_scene_and_datums", fake_prepare)
    cfg = OmegaConf.create(
        {
            "data": {"name": "sdss"},
            "trainer": {},
            "physics": {"bandpasses": [], "held_out_band": None},
            "calibration": {},
        }
    )

    with pytest.raises(ExpectedStop):
        train._build_trainer_components(
            cfg,
            paths={"registry_path": tmp_path / "registry.yaml"},
            device=torch.device("cpu"),
            precision_policy="portable_fp32",
            require_prepared_cache=True,
        )


def test_prepared_cache_staging_is_atomic_and_reuses_matching_copy(tmp_path, monkeypatch):
    import train

    source = tmp_path / "source.pt"
    target = tmp_path / "cache" / "target.pt"
    source.write_bytes(b"schema-2-cache")

    size = train._stage_prepared_cache(source, target)
    assert size == len(b"schema-2-cache")
    assert target.read_bytes() == b"schema-2-cache"
    assert target.stat().st_mtime_ns == source.stat().st_mtime_ns

    monkeypatch.setattr(
        train.shutil,
        "copy2",
        lambda *_args, **_kwargs: pytest.fail("matching staged cache was copied again"),
    )
    assert train._stage_prepared_cache(source, target) == size


def test_large_arrays_externalize_to_mmap_sidecars(tmp_path):
    import numpy as np
    import torch

    from survey_fusion.data import prepared as prepared_mod

    big_numpy = np.arange(300_000, dtype=np.float64).reshape(600, 500)
    big_torch = torch.arange(300_000, dtype=torch.float32)
    payload_scene = {"big": big_numpy, "small": np.ones(3)}
    payload_store = {"tensor": big_torch, "note": "x"}
    path = tmp_path / "prepared_deadbeef"
    prepared_mod.save_prepared(path, payload_scene, payload_store)

    sidecars = sorted((path / "arrays").glob("*.npy"))
    assert len(sidecars) == 2
    assert (path / "meta.pkl").stat().st_size < 10_000

    scene, store = prepared_mod.load_prepared(path)
    assert isinstance(scene["big"], np.ndarray)
    assert scene["big"].base is not None or isinstance(scene["big"], np.memmap)
    np.testing.assert_array_equal(np.asarray(scene["big"]), big_numpy)
    assert torch.equal(store["tensor"], big_torch)
    # copy-on-write: writing to the loaded array must not corrupt the cache
    np.asarray(scene["big"])[0, 0] = -1.0
    scene2, _ = prepared_mod.load_prepared(path)
    assert float(np.asarray(scene2["big"])[0, 0]) == 0.0


def test_prepare_seed_defaults_to_42_and_is_overridable():
    from survey_fusion.data.prepared import prepare_seed

    assert prepare_seed({}) == 42
    assert prepare_seed(None) == 42
    assert prepare_seed({"prepare_seed": 7}) == 7
    for bad in (None, True):
        with pytest.raises(ValueError, match="prepare_seed"):
            prepare_seed({"prepare_seed": bad})


def test_store_identity_keys_do_not_change_the_store_digest(tmp_path):
    base = {"observation_root": str(tmp_path), "spec_min_ivar": 0.0}
    tagged = {**base, "prepare_seed": 7, "expected_prepared_digest": "f" * 64}
    kwargs = dict(seed=42, bands=["r"], manifest_path=None)
    assert prepared_digest(base, **kwargs) == prepared_digest(tagged, **kwargs)
    assert prepared_digest(base, **kwargs) != prepared_digest(base, **{**kwargs, "seed": 7})


def test_build_components_passes_the_data_prepare_seed_not_the_run_seed(monkeypatch, tmp_path):
    import train

    class ExpectedStop(RuntimeError):
        pass

    def fake_prepare(*_args, seed, **_kwargs):
        assert seed == 42
        raise ExpectedStop

    monkeypatch.setattr(train, "load_bandpasses", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(train, "prepare_scene_and_datums", fake_prepare)
    cfg = OmegaConf.create(
        {
            "data": {"name": "sdss"},
            "trainer": {},
            "physics": {"bandpasses": [], "held_out_band": None},
            "calibration": {},
            "run": {"seed": 7},
        }
    )
    with pytest.raises(ExpectedStop):
        train._build_trainer_components(
            cfg, paths={"registry_path": tmp_path / "registry.yaml"},
            device=torch.device("cpu"), precision_policy="portable_fp32",
        )


def _keyed_cfg(tmp_path, **extra):
    return {"prepared_dir": str(tmp_path / "prepared"), "observation_root": str(tmp_path), **extra}


def test_expected_digest_mismatch_stops_before_any_build(tmp_path, monkeypatch):
    import train

    monkeypatch.setattr(train, "load_sdss_scene", lambda *_a, **_k: pytest.fail("built a store"))
    with pytest.raises(RuntimeError, match="Prepared store mismatch"):
        train.prepare_scene_and_datums(
            _keyed_cfg(tmp_path, expected_prepared_digest="0" * 64), _bandpasses(),
            trainer_cfg={}, seed=42, device=torch.device("cpu"),
        )


def test_matching_expected_digest_proceeds_to_the_store(tmp_path, monkeypatch):
    import train

    class Built(RuntimeError):
        pass

    def fake_build(*_a, **_k):
        raise Built

    cfg = _keyed_cfg(tmp_path)
    digest = prepared_digest(cfg, seed=42, bands=list(_bandpasses().keys()), manifest_path=None)
    monkeypatch.setattr(train, "load_sdss_scene", fake_build)
    with pytest.raises(Built):
        train.prepare_scene_and_datums(
            {**cfg, "expected_prepared_digest": digest}, _bandpasses(),
            trainer_cfg={}, seed=42, device=torch.device("cpu"),
        )


def test_expected_digest_without_a_cache_is_refused(tmp_path):
    import train

    with pytest.raises(ValueError, match="expected_prepared_digest needs the prepared cache"):
        train.prepare_scene_and_datums(
            {"prepared_cache": False, "expected_prepared_digest": "0" * 64}, _bandpasses(),
            trainer_cfg={}, seed=42, device=torch.device("cpu"),
        )


def test_train_raises_the_open_file_soft_limit_for_the_store(monkeypatch):
    import resource

    import train

    calls = []
    monkeypatch.setattr(resource, "getrlimit", lambda kind: (1024, 1048576))
    monkeypatch.setattr(resource, "setrlimit", lambda kind, limits: calls.append(limits))
    train._raise_open_file_limit()
    assert calls == [(train.OPEN_FILE_LIMIT, 1048576)]
    calls.clear()
    monkeypatch.setattr(resource, "getrlimit", lambda kind: (1048576, 1048576))
    train._raise_open_file_limit()
    assert calls == []


def test_build_components_requires_the_store_when_its_digest_is_pinned(monkeypatch, tmp_path):
    import train

    class ExpectedStop(RuntimeError):
        pass

    def fake_prepare(*_args, require_prepared_cache=False, **_kwargs):
        assert require_prepared_cache is True
        raise ExpectedStop

    monkeypatch.setattr(train, "load_bandpasses", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(train, "prepare_scene_and_datums", fake_prepare)
    cfg = OmegaConf.create({
        "data": {"name": "sdss", "expected_prepared_digest": "0" * 64}, "trainer": {},
        "physics": {"bandpasses": [], "held_out_band": None}, "calibration": {},
    })
    with pytest.raises(ExpectedStop):
        train._build_trainer_components(
            cfg, paths={"registry_path": tmp_path / "registry.yaml"},
            device=torch.device("cpu"), precision_policy="portable_fp32",
        )


def test_build_components_refuses_unfilled_values(monkeypatch, tmp_path):
    import train

    monkeypatch.setattr(train, "load_bandpasses", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(train, "prepare_scene_and_datums", lambda *_a, **_k: pytest.fail("reached the store"))
    cfg = OmegaConf.create({
        "data": {"name": "sdss", "prepared_dir": "???"}, "trainer": {},
        "physics": {"bandpasses": [], "held_out_band": None}, "calibration": {},
    })
    with pytest.raises(Exception, match="Missing mandatory value"):
        train._build_trainer_components(
            cfg, paths={"registry_path": tmp_path / "registry.yaml"},
            device=torch.device("cpu"), precision_policy="portable_fp32",
        )
