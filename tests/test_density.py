import pytest

from crowd.config import load_config
from crowd.density import DensityEstimator
from crowd.media import read_first_frame, read_image
from helpers import MALL_IMAGE, VIDEO3, needs_demo_data, needs_density_model

pytestmark = needs_demo_data


@pytest.fixture(scope="module")
def estimator():
    return DensityEstimator(load_config()["density_weights"])


def test_missing_weights_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing.pth"):
        DensityEstimator(tmp_path / "missing.pth")


@needs_density_model
def test_predict_mall_frame(estimator):
    density, count = estimator.predict(read_image(MALL_IMAGE))
    assert density.shape == (480 // 8, 640 // 8)
    assert 5 < count < 120
    assert abs(float(density.sum()) - count) < 1e-3


@needs_density_model
def test_predict_uses_imagenet_normalization(estimator):
    # Old code skipped Normalize and returned ~403 on this frame; normalized input gives ~503.
    _, count = estimator.predict(read_first_frame(VIDEO3))
    assert 450 < count < 560
