import pytest

from crowd.config import load_config
from crowd.detector import PersonDetector
from crowd.media import read_image
from helpers import MALL_IMAGE, needs_demo_data, needs_detector_model

pytestmark = needs_demo_data


@pytest.fixture(scope="module")
def detector():
    cfg = load_config()
    return PersonDetector(cfg["detector_weights"], cfg["detector_conf"])


def test_missing_weights_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="nothing.pt"):
        PersonDetector(tmp_path / "nothing.pt")


@needs_detector_model
def test_detects_people_on_mall_frame(detector):
    boxes = detector.detect(read_image(MALL_IMAGE))
    assert len(boxes) >= 5
    for x1, y1, x2, y2, score in boxes:
        assert 0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480
        assert score >= 0.35
