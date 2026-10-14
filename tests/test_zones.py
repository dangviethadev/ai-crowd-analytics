import numpy as np
import pytest

from crowd import zones as zn

SQUARE = [(10, 20), (110, 20), (110, 220), (10, 220)]


def test_parse_old_box_format_becomes_polygon():
    data = {"drawList": [{"type": "box", "xy": {"x": 519, "width": 1128, "y": 489, "height": 556}}]}
    assert zn.parse_zones(data) == [
        {"name": "Zone 1", "points": [(519, 489), (1647, 489), (1647, 1045), (519, 1045)]}]


def test_parse_polygon_format():
    data = {"drawList": [{"type": "polygon", "name": "Gate", "points": [[0, 0], [50, 0], [25, 40]]}]}
    assert zn.parse_zones(data) == [{"name": "Gate", "points": [(0, 0), (50, 0), (25, 40)]}]


def test_save_then_load_roundtrip(tmp_path):
    media = tmp_path / "clip.mp4"
    zones = [{"name": "Entrance", "points": [(10, 20), (110, 20), (60, 220)]}]
    assert zn.save_zones(media, zones) == tmp_path / "clip.json"
    assert zn.load_zones(media) == zones


def test_load_zones_missing_file_returns_empty(tmp_path):
    assert zn.load_zones(tmp_path / "none.jpg") == []


def test_dotted_filename_keeps_full_stem(tmp_path):
    media = tmp_path / "cam.01.mp4"
    zn.save_zones(media, [{"name": "A", "points": SQUARE}])
    assert (tmp_path / "cam.01.json").exists()
    assert zn.load_zones(media)[0]["name"] == "A"


def test_clamp_polygon_clips_to_image_and_drops_duplicates():
    points = [(-10, -10), (150, -5), (150, 120), (150, 120), (-3, 120)]
    assert zn.clamp_polygon(points, 100, 80) == [(0, 0), (100, 0), (100, 80), (0, 80)]


def test_clamp_polygon_rejects_degenerate():
    assert zn.clamp_polygon([(0, 0), (50, 50)], 100, 100) is None            # fewer than 3 points
    assert zn.clamp_polygon([(0, 0), (50, 50), (100, 100)], 100, 100) is None  # a line, no area
    assert zn.clamp_polygon([(200, 200), (300, 200), (300, 300)], 100, 100) is None  # outside


def test_fit_zones_drops_invalid():
    zones = [{"name": "A", "points": [(0, 0), (10, 0), (10, 10)]},
             {"name": "B", "points": [(500, 500), (600, 500), (600, 600)]}]
    assert zn.fit_zones(zones, 100, 100) == [{"name": "A", "points": [(0, 0), (10, 0), (10, 10)]}]


def test_density_zone_counts_scales_to_map():
    density = np.zeros((10, 20))
    density[0:5, 0:10] = 1.0  # top-left quarter of a 160x80 image holds 50 people
    zones = [{"name": "TL", "points": [(0, 0), (80, 0), (80, 40), (0, 40)]},
             {"name": "All", "points": [(0, 0), (160, 0), (160, 80), (0, 80)]}]
    counts = zn.density_zone_counts(density, zones, 160, 80)
    assert [name for name, _ in counts] == ["TL", "All"]
    assert counts[0][1] == pytest.approx(50.0) and counts[1][1] == pytest.approx(50.0)


def test_density_zone_counts_triangle_covers_half():
    density = np.ones((10, 10))  # 100 people spread evenly over an 80x80 image
    triangle = [{"name": "T", "points": [(0, 0), (80, 0), (0, 80)]}]
    assert zn.density_zone_counts(density, triangle, 80, 80)[0][1] == pytest.approx(50.0, abs=3)


def test_point_zone_counts_in_polygon():
    triangle = {"name": "T", "points": [(0, 0), (100, 0), (0, 100)]}
    square = {"name": "S", "points": [(60, 60), (100, 60), (100, 100), (60, 100)]}
    points = [(10, 10), (80, 80), (90, 90), (70, 20)]
    assert zn.point_zone_counts(points, [triangle, square]) == [("T", 2), ("S", 2)]
