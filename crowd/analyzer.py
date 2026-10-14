"""Single entry point for both UIs: a frame goes in, a display-ready result comes out."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from crowd.density import DensityEstimator
from crowd.detector import PersonDetector
from crowd.media import iter_frames
from crowd.render import colorize_density, compose
from crowd.zones import density_zone_counts, fit_zones, point_zone_counts

DENSITY = "density"
DETECTION = "detection"
MIN_SIDE = 32  # the density network needs at least a few stride-16 cells


def density_level(count, low, high):
    if count < low:
        return "Low"
    if count < high:
        return "Medium"
    return "High"


def resize_for_model(frame, max_side):
    """Shrink so the longer side <= max_side; enlarge tiny images. Returns (image, scale)."""
    h, w = frame.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if min(h, w) * scale < MIN_SIDE:
        scale = MIN_SIDE / min(h, w)
    if scale == 1.0:
        return frame, 1.0
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(frame, size, interpolation=interp), scale


class Analyzer:
    def __init__(self, cfg):
        self.cfg = cfg
        self._density = None
        self._detector = None

    @property
    def density_available(self):
        return Path(self.cfg["density_weights"]).exists()

    @property
    def density(self):
        # Loaded on first use so the app still starts (in Detection mode) without the density model.
        if self._density is None:
            self._density = DensityEstimator(self.cfg["density_weights"])
        return self._density

    @property
    def detector(self):
        # Loaded on first use so Density-only sessions start faster.
        if self._detector is None:
            self._detector = PersonDetector(self.cfg["detector_weights"], self.cfg["detector_conf"])
        return self._detector

    def process(self, frame, mode=DENSITY, zones=()):
        h, w = frame.shape[:2]
        zones = fit_zones(zones, w, h)
        small, scale = resize_for_model(frame, self.cfg["max_side"])
        boxes, heatmap = [], None
        if mode == DETECTION:
            for x1, y1, x2, y2, score in self.detector.detect(small):
                boxes.append((int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale), score))
            count = len(boxes)
            feet = [((x1 + x2) / 2, y2) for x1, _, x2, y2, _ in boxes]
            zone_counts = point_zone_counts(feet, zones)
        else:
            density, count = self.density.predict(small)
            zone_counts = density_zone_counts(density, zones, w, h)
            heatmap = colorize_density(density, (small.shape[1], small.shape[0]))
        thresholds = self.cfg["level_thresholds"]
        result = {
            "mode": mode,
            "count": int(round(count)),
            "zones": zones,
            "zone_counts": [(name, int(round(c))) for name, c in zone_counts],
            "boxes": boxes,
            "heatmap": heatmap,
            "level": density_level(count, thresholds["low"], thresholds["high"]),
            "fresh": True,
        }
        result["overlay"] = compose(frame, result)
        return result

    def stream(self, cap, mode=DENSITY, zones=(), realtime=True):
        """Yield a result for every displayed frame.

        The model runs in a background thread at most once per interval, so playback does not
        freeze during inference. Frames in between reuse the latest result ("fresh" is False).
        """
        interval = self.cfg[f"{mode}_interval_sec"]
        pool = ThreadPoolExecutor(max_workers=1)
        pending, last, last_ts = None, None, None
        try:
            for ts, frame in iter_frames(cap, realtime):
                fresh = False
                if pending is not None and pending.done():
                    last, pending, fresh = pending.result(), None, True
                if pending is None and (last_ts is None or ts - last_ts >= interval):
                    pending, last_ts = pool.submit(self.process, frame, mode, zones), ts
                if last is None:  # nothing to show before the first result
                    last, pending, fresh = pending.result(), None, True
                yield {**last, "overlay": compose(frame, last), "fresh": fresh}
        finally:
            pool.shutdown(wait=False)
