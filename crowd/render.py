"""Drawing helpers. Images are BGR numpy arrays."""
import cv2
import numpy as np

from crowd.zones import label_point

PERSON_COLOR = (0, 230, 0)
# Distinct RGB colors so neighbouring or overlapping zones are easy to tell apart.
ZONE_COLORS = [(230, 25, 75), (0, 130, 200), (255, 150, 0), (145, 30, 180), (0, 170, 140), (240, 50, 230)]


def zone_color(index, bgr=True):
    r, g, b = ZONE_COLORS[index % len(ZONE_COLORS)]
    return (b, g, r) if bgr else (r, g, b)


def colorize_density(density, size):
    """JET heatmap of a density map, resized to size=(width, height)."""
    peak = float(density.max())
    norm = density / peak if peak > 0 else density
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.resize(heat, size, interpolation=cv2.INTER_LINEAR)


def _label(img, text, center, color, scale):
    """Text centered on a point, with a dark outline so it reads on any background."""
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    org = (int(center[0] - w / 2), int(center[1] + h / 2))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def compose(frame, result):
    """Copy of frame with person boxes, zones and per-zone counts drawn on it."""
    out = frame.copy()
    thick = max(2, frame.shape[1] // 640)
    scale = max(0.6, frame.shape[1] / 1600)
    for x1, y1, x2, y2, _ in result["boxes"]:
        cv2.rectangle(out, (x1, y1), (x2, y2), PERSON_COLOR, thick)
    for i, (zone, (_, count)) in enumerate(zip(result["zones"], result["zone_counts"])):
        color = zone_color(i)
        cv2.polylines(out, [np.array(zone["points"], np.int32)], True, color, thick)
        _label(out, f"{zone['name']}: {count}", label_point(zone["points"]), color, scale)
    return out
