import time

import cv2
import numpy as np

from crowd import media
from helpers import FakeCap


def _frames(n):
    return [np.full((4, 4, 3), i, np.uint8) for i in range(n)]


def test_read_image_non_ascii_path(tmp_path):
    folder = tmp_path / "ảnh thử"
    folder.mkdir()
    path = folder / "khung_hình.png"
    cv2.imencode(".png", np.zeros((8, 8, 3), np.uint8))[1].tofile(str(path))
    image = media.read_image(path)
    assert image is not None and image.shape == (8, 8, 3)


def test_read_image_missing_or_invalid(tmp_path):
    assert media.read_image(tmp_path / "missing.jpg") is None
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    assert media.read_image(bad) is None


def test_decode_image_bytes():
    data = cv2.imencode(".jpg", np.zeros((6, 9, 3), np.uint8))[1].tobytes()
    assert media.decode_image(data).shape == (6, 9, 3)
    assert media.decode_image(b"") is None


def test_open_video_missing_file_returns_none(tmp_path):
    assert media.open_video(str(tmp_path / "missing.mp4")) is None


def test_video_fps_falls_back_when_invalid():
    assert media.video_fps(FakeCap([], 0)) == media.DEFAULT_FPS
    assert media.video_fps(FakeCap([], 1000)) == media.DEFAULT_FPS
    assert media.video_fps(FakeCap([], 30)) == 30


def test_iter_frames_without_realtime_yields_every_frame():
    out = list(media.iter_frames(FakeCap(_frames(5), 10), realtime=False))
    assert [ts for ts, _ in out] == [0.0, 0.1, 0.2, 0.3, 0.4]


def test_iter_frames_realtime_drops_frames_for_slow_consumer():
    seen = 0
    for _ in media.iter_frames(FakeCap(_frames(30), 100), realtime=True):
        seen += 1
        time.sleep(0.03)
    assert 0 < seen < 30
