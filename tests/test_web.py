import shutil
from types import SimpleNamespace

import gradio as gr
import pytest

from crowd.zones import load_zones
from helpers import MALL_IMAGE, VIDEO3, needs_demo_data, needs_density_model, needs_detector_model
from web import app

pytestmark = needs_demo_data
DEMO_IMAGE = "images/seq_000002.jpg"


def click(x, y):
    return SimpleNamespace(index=[x, y])  # stands in for gr.SelectData


@pytest.fixture
def upload(tmp_path):
    image = tmp_path / "mall.jpg"
    shutil.copy(MALL_IMAGE, image)
    return str(image)


def test_ui_builds():
    assert isinstance(app.build_ui(), gr.Blocks)


def test_demo_files_lists_images_and_videos():
    files = app.demo_files()
    assert DEMO_IMAGE in files and "videos/Video3.avi" in files


@needs_density_model
def test_analyze_image_density():
    overlay, heatmap, stats, table = next(app.analyze("Demo file", DEMO_IMAGE, None, "Density"))
    assert overlay.shape == (480, 640, 3) and heatmap is not None
    assert "People" in stats and any(level in stats for level in ("Low", "Medium", "High"))


@needs_detector_model
def test_analyze_image_detection_has_no_heatmap():
    _, heatmap, stats, _ = next(app.analyze("Demo file", DEMO_IMAGE, None, "Detection"))
    assert heatmap is None and "Density mode" in stats


def test_analyze_without_file_raises():
    with pytest.raises(gr.Error):
        next(app.analyze("Upload", DEMO_IMAGE, None, "Density"))


@needs_detector_model
def test_analyze_video_streams_frames_with_zone_table():
    results = app.analyze("Demo file", "videos/Video3.avi", None, "Detection")
    first, second = next(results), next(results)
    results.close()
    assert first[0].shape[1] <= app.DISPLAY_WIDTH
    assert len(first[3]) == len(load_zones(VIDEO3))


@needs_detector_model
def test_zone_drawing_flow_saves_next_to_file(upload):
    state, canvas, *_ = app.zones_load("Upload", None, upload)
    assert canvas.shape == (480, 640, 3)
    for x, y in [(10, 10), (300, 20), (999, 999)]:
        state, *_ = app.zones_click(state, click(x, y))
    state, *_ = app.zones_undo(state)
    state, *_ = app.zones_click(state, click(150, 300))
    state, *_ = app.zones_add(state, "Gate")
    assert state["points"] == [] and state["zones"][0]["name"] == "Gate"
    state, *_ = app.zones_save(state)
    assert load_zones(upload) == [{"name": "Gate", "points": [(10, 10), (300, 20), (150, 300)]}]
    # The analysis tab picks up the saved zones for the same file.
    _, _, _, table = next(app.analyze("Upload", None, upload, "Detection"))
    assert table[0][0] == "Gate"


def test_zone_add_needs_three_points_and_names_default(upload):
    state, *_ = app.zones_load("Upload", None, upload)
    state, *_ = app.zones_click(state, click(10, 10))
    state, _, _, _, status = app.zones_add(state, "")
    assert state["zones"] == [] and "3 points" in status
    for x, y in [(200, 10), (100, 200)]:
        state, *_ = app.zones_click(state, click(x, y))
    state, *_ = app.zones_add(state, "")
    assert state["zones"][0]["name"] == "Zone 1"


def test_zone_delete(upload):
    state, *_ = app.zones_load("Upload", None, upload)
    for name in ("A", "B"):
        for x, y in [(10, 10), (200, 10), (100, 200)]:
            state, *_ = app.zones_click(state, click(x, y))
        state, *_ = app.zones_add(state, name)
    state, *_ = app.zones_delete(state, "A")
    assert [z["name"] for z in state["zones"]] == ["B"]


def test_zone_click_before_load_is_ignored():
    state, canvas, *_ = app.zones_click(app.empty_zone_state(), click(5, 5))
    assert state["points"] == [] and canvas is None
