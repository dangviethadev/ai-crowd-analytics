import time

import cv2
import numpy as np
import pytest

from crowd.analyzer import DENSITY, DETECTION, Analyzer, density_level, resize_for_model
from crowd.config import load_config
from crowd.media import read_image
from helpers import MALL_IMAGE, FakeCap, needs_density_model, needs_detector_model


@pytest.fixture(scope="module")
def analyzer():
    return Analyzer(load_config())


@pytest.fixture(scope="module")
def mall():
    if not MALL_IMAGE.exists():
        pytest.skip("demo data not found in data/demo (see README step 5)")
    return read_image(MALL_IMAGE)


def test_density_level():
    assert density_level(5, 20, 100) == "Low"
    assert density_level(20, 20, 100) == "Medium"
    assert density_level(100, 20, 100) == "High"


def test_resize_for_model_shrinks_large_and_enlarges_tiny():
    small, scale = resize_for_model(np.zeros((2160, 3840, 3), np.uint8), 1280)
    assert small.shape[:2] == (720, 1280) and scale == pytest.approx(1 / 3)
    up, _ = resize_for_model(np.zeros((10, 20, 3), np.uint8), 1280)
    assert min(up.shape[:2]) == 32
    same, s = resize_for_model(np.zeros((480, 640, 3), np.uint8), 1280)
    assert same.shape[:2] == (480, 640) and s == 1.0


@needs_density_model
def test_process_density_with_zones(analyzer, mall):
    zones = [{"name": "Left", "points": [(0, 0), (320, 0), (320, 480), (0, 480)]},
             {"name": "Right", "points": [(320, 0), (640, 0), (640, 480), (320, 480)]}]
    r = analyzer.process(mall, DENSITY, zones)
    assert r["heatmap"] is not None and r["boxes"] == []
    assert r["overlay"].shape == mall.shape
    assert abs(r["zone_counts"][0][1] + r["zone_counts"][1][1] - r["count"]) <= 2
    assert r["level"] in {"Low", "Medium", "High"} and r["fresh"]


@needs_detector_model
def test_process_detection_has_boxes_and_no_heatmap(analyzer, mall):
    r = analyzer.process(mall, DETECTION)
    assert r["heatmap"] is None
    assert r["count"] == len(r["boxes"]) >= 5


@needs_detector_model
def test_detection_boxes_are_in_original_coordinates(analyzer, mall):
    big = cv2.resize(mall, (2560, 1920))  # shrunk to 1280 wide before YOLO
    r = analyzer.process(big, DETECTION)
    assert max(b[2] for b in r["boxes"]) > 1280


@needs_density_model
def test_tiny_image_does_not_crash(analyzer):
    r = analyzer.process(np.zeros((10, 10, 3), np.uint8), DENSITY)
    assert r["count"] >= 0


@needs_detector_model
def test_out_of_bounds_zone_is_ignored(analyzer, mall):
    r = analyzer.process(mall, DETECTION, [{"name": "Far", "points": [(5000, 5000), (6000, 5000), (6000, 6000)]}])
    assert r["zone_counts"] == []


@needs_density_model
def test_stream_runs_model_once_per_interval(analyzer, mall, monkeypatch):
    calls = []
    real_process = analyzer.process
    monkeypatch.setattr(analyzer, "process", lambda *a, **k: calls.append(1) or real_process(*a, **k))
    cap = FakeCap([mall] * 25, fps=10)  # 2.5 s of video, density interval 1.0 s
    results = list(analyzer.stream(cap, DENSITY, realtime=True))
    assert len(results) >= 20
    assert len(calls) == 3
    assert results[0]["fresh"]


@needs_detector_model
def test_stream_keeps_playing_while_model_is_busy(analyzer, mall, monkeypatch):
    first = analyzer.process(mall, DETECTION)

    def slow_process(*args, **kwargs):
        time.sleep(0.5)
        return first

    monkeypatch.setattr(analyzer, "process", slow_process)
    cap = FakeCap([mall] * 30, fps=10)  # 3 s of video; a model run takes 0.5 s
    results = list(analyzer.stream(cap, DENSITY, realtime=True))
    # Only the frames before the first result are dropped (~4); blocking inference kept just 18.
    assert len(results) >= 24
