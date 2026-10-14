"""Image and video input helpers."""
import time

import cv2
import numpy as np

DEFAULT_FPS = 25.0


def read_image(path):
    """Like cv2.imread, but also works with non-ASCII paths on Windows. Returns None on failure."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return decode_image(data.tobytes())


def decode_image(data):
    """Decode image bytes (e.g. a web upload) to a BGR array, or None."""
    arr = np.frombuffer(data, dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def open_video(source):
    """Open a video file, camera index or stream URL. Returns None if it cannot be opened."""
    if isinstance(source, str) and source.strip().isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def read_first_frame(path):
    cap = open_video(path)
    if cap is None:
        return None
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def video_fps(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps if 0 < fps <= 120 else DEFAULT_FPS


def iter_frames(cap, realtime=True):
    """Yield (timestamp_sec, frame).

    realtime=True is for video files: sleep when ahead of the clock, drop frames when behind.
    Use realtime=False for cameras/streams, which already deliver frames at live speed.
    """
    fps = video_fps(cap)
    start = time.perf_counter()
    idx = -1
    while True:
        idx += 1
        ts = idx / fps
        if realtime:
            lag = (time.perf_counter() - start) - ts
            if lag < 0:
                time.sleep(-lag)
            elif lag > 1.0 / fps:
                if not cap.grab():
                    return
                continue
        ok, frame = cap.read()
        if not ok:
            return
        yield ts, frame
