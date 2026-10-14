from pathlib import Path

import cv2
import pytest

ROOT = Path(__file__).resolve().parent.parent
MALL_IMAGE = ROOT / "data" / "demo" / "images" / "seq_000002.jpg"
VIDEO3 = ROOT / "data" / "demo" / "videos" / "Video3.avi"

DENSITY_WEIGHTS = ROOT / "weights" / "crowd_vgg19.pth"
DETECTOR_WEIGHTS = ROOT / "weights" / "yolov8n.pt"

# Demo data is local only (not in git), so tests that use it are skipped when it is missing.
needs_demo_data = pytest.mark.skipif(not (MALL_IMAGE.exists() and VIDEO3.exists()),
                                     reason="demo data not found in data/demo (see README step 5)")
# The density model file is delivered separately; Density tests are skipped until it is installed.
needs_density_model = pytest.mark.skipif(not DENSITY_WEIGHTS.exists(),
                                         reason="density model weights/crowd_vgg19.pth not installed")
# Model files are not in git; see weights/README.md to download yolov8n.pt.
needs_detector_model = pytest.mark.skipif(not DETECTOR_WEIGHTS.exists(),
                                          reason="detector model weights/yolov8n.pt not downloaded")


class FakeCap:
    """Minimal stand-in for cv2.VideoCapture."""

    def __init__(self, frames, fps):
        self.frames = list(frames)
        self.fps = fps
        self.pos = 0

    def get(self, prop):
        return self.fps if prop == cv2.CAP_PROP_FPS else 0

    def read(self):
        if self.pos >= len(self.frames):
            return False, None
        self.pos += 1
        return True, self.frames[self.pos - 1]

    def grab(self):
        return self.read()[0]

    def release(self):
        pass
