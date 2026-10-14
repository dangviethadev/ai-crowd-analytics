"""Without weights/crowd_vgg19.pth the apps must still start and run Detection mode."""
import pytest

from crowd.analyzer import DENSITY, DETECTION, Analyzer
from crowd.config import load_config
from crowd.media import read_image
from helpers import MALL_IMAGE, needs_demo_data, needs_detector_model

pytestmark = needs_demo_data


@pytest.fixture
def cfg_without_density(tmp_path):
    cfg = load_config()
    cfg["density_weights"] = str(tmp_path / "crowd_vgg19.pth")
    return cfg


@needs_detector_model
def test_analyzer_starts_and_detection_works(cfg_without_density):
    analyzer = Analyzer(cfg_without_density)
    assert not analyzer.density_available
    result = analyzer.process(read_image(MALL_IMAGE), DETECTION)
    assert result["count"] == len(result["boxes"])


def test_density_mode_explains_missing_model(cfg_without_density):
    analyzer = Analyzer(cfg_without_density)
    with pytest.raises(FileNotFoundError, match="Density model is not installed"):
        analyzer.process(read_image(MALL_IMAGE), DENSITY)


def test_desktop_starts_in_detection_mode(qapp, cfg_without_density):
    from desktop.main_window import MainWindow
    window = MainWindow(cfg_without_density)
    assert window.mode == DETECTION
    assert "Detection" in window.statusbar.currentMessage()


@needs_detector_model
def test_web_density_shows_friendly_error(cfg_without_density, monkeypatch):
    import gradio as gr
    from web import app
    monkeypatch.setattr(app, "_analyzer", Analyzer(cfg_without_density))
    with pytest.raises(gr.Error, match="Density model is not installed"):
        next(app.analyze("Demo file", "images/seq_000002.jpg", None, "Density"))
    _, heatmap, _, _ = next(app.analyze("Demo file", "images/seq_000002.jpg", None, "Detection"))
    assert heatmap is None
