from omegaconf import OmegaConf

from survey_fusion.utils.paths import dump_config


def test_dump_config_resolves_interpolations(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SURVEY_FUSION_TEST_CONFIG_VALUE", "resolved-value")
    config = OmegaConf.create(
        {"value": "${oc.env:SURVEY_FUSION_TEST_CONFIG_VALUE}"}
    )
    path = tmp_path / "config.yaml"

    dump_config(config, path)

    saved = path.read_text(encoding="utf-8")
    assert "${" not in saved
    assert OmegaConf.load(path).value == "resolved-value"
