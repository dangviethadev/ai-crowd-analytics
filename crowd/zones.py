"""Zones are polygons drawn in the Setup tool and stored next to the media file as JSON."""
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

MIN_AREA = 4.0  # pixels^2; smaller polygons are treated as empty


def zone_path_for(media_path):
    """Zones for 'clip.mp4' live in 'clip.json'."""
    return Path(media_path).with_suffix(".json")


def parse_zones(data):
    """Read polygons; old rectangle zones ("xy" boxes) become 4-point polygons."""
    zones = []
    for i, item in enumerate(data.get("drawList", [])):
        if "points" in item:
            points = [(int(x), int(y)) for x, y in item["points"]]
        else:
            xy = item["xy"]
            x1, y1 = int(xy["x"]), int(xy["y"])
            x2, y2 = x1 + int(xy["width"]), y1 + int(xy["height"])
            points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        zones.append({"name": item.get("name") or f"Zone {i + 1}", "points": points})
    return zones


def load_zones(media_path):
    path = zone_path_for(media_path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return parse_zones(json.load(f))


def save_zones(media_path, zones):
    draw_list = [{"type": "polygon", "name": z["name"], "points": [list(p) for p in z["points"]]} for z in zones]
    data = {"imagePath": Path(media_path).name, "drawList": draw_list, "createdTime": str(datetime.now())}
    path = zone_path_for(media_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return path


def clamp_polygon(points, width, height):
    """Clip points to the image and drop repeated points. Returns None if no area is left."""
    clipped = []
    for x, y in points:
        p = (int(max(0, min(x, width))), int(max(0, min(y, height))))
        if not clipped or p != clipped[-1]:
            clipped.append(p)
    if len(clipped) > 1 and clipped[0] == clipped[-1]:
        clipped.pop()
    if len(clipped) < 3 or cv2.contourArea(np.array(clipped, np.float32)) < MIN_AREA:
        return None
    return clipped


def fit_zones(zones, width, height):
    fitted = []
    for zone in zones:
        points = clamp_polygon(zone["points"], width, height)
        if points:
            fitted.append({"name": zone["name"], "points": points})
    return fitted


def label_point(points):
    """Where to put a zone's name: the polygon centroid (inside for any convex zone)."""
    m = cv2.moments(np.array(points, np.float32))
    if m["m00"]:
        return int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
    pts = np.array(points, np.float64)
    return int(pts[:, 0].mean()), int(pts[:, 1].mean())


def _coverage(points, width, height, out_w, out_h, upsample=8):
    """Fraction of each density-map cell covered by the polygon (drawn at higher resolution)."""
    scale = np.array([out_w * upsample / width, out_h * upsample / height])
    pts = np.round(np.array(points, np.float64) * scale).astype(np.int32)
    mask = np.zeros((out_h * upsample, out_w * upsample), np.float32)
    cv2.fillPoly(mask, [pts], 1.0)
    return cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_AREA)


def density_zone_counts(density, zones, width, height):
    """Sum the density map inside each zone. Zones are in image pixels; the map is smaller."""
    out_h, out_w = density.shape
    return [(z["name"], float((density * _coverage(z["points"], width, height, out_w, out_h)).sum()))
            for z in zones]


def point_zone_counts(points, zones):
    """Count points (x, y) inside each polygon (edges count as inside)."""
    counts = []
    for zone in zones:
        poly = np.array(zone["points"], np.float32)
        inside = sum(1 for px, py in points if cv2.pointPolygonTest(poly, (float(px), float(py)), False) >= 0)
        counts.append((zone["name"], inside))
    return counts
