import numpy as np

from survey_fusion.evaluation.visual_selection import select_visual_panel_centres


def test_visual_selection_is_data_only_and_deterministic() -> None:
    reference = np.zeros((32, 32), dtype=np.float64)
    reference[5, 5] = 20.0
    reference[20, 20] = 10.0
    reference[20, 22] = 9.0
    reference[22, 20] = 8.0
    reference[8, 24] = 4.0
    reference[10, 24] = 3.5
    reference[10, 26] = 3.0

    first = select_visual_panel_centres(
        reference,
        native_shape=(128, 128),
        downsample=4,
    )
    second = select_visual_panel_centres(
        reference,
        native_shape=(128, 128),
        downsample=4,
    )

    assert first == second
    assert [item["category"] for item in first[:5]] == [
        "center",
        "southwest_quadrant",
        "southeast_quadrant",
        "northwest_quadrant",
        "northeast_quadrant",
    ]
    assert {item["category"] for item in first} >= {
        "bright_isolated",
        "crowded_blend",
        "faint_sources",
    }


def test_visual_selection_rejects_empty_reference() -> None:
    try:
        select_visual_panel_centres(
            np.full((4, 4), np.nan),
            native_shape=(16, 16),
            downsample=4,
        )
    except ValueError as error:
        assert "no finite pixels" in str(error)
    else:
        raise AssertionError("Expected an empty reference to fail.")


def test_visual_selection_respects_native_cutout_margin() -> None:
    reference = np.zeros((64, 64), dtype=np.float64)
    reference[2, 2] = 100.0
    reference[20, 20] = 20.0
    reference[20, 22] = 18.0
    reference[22, 20] = 16.0
    reference[42, 42] = 10.0
    reference[42, 44] = 8.0
    reference[44, 42] = 6.0

    selections = select_visual_panel_centres(
        reference,
        native_shape=(512, 512),
        downsample=8,
        native_margin=96,
    )

    for selection in selections[5:]:
        assert 96 <= int(selection["x"]) <= 512 - 96
        assert 96 <= int(selection["y"]) <= 512 - 96
