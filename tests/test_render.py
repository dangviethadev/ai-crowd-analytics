import numpy as np

from crowd.render import colorize_density, compose, zone_color


def test_colorize_density_hot_is_red_and_resized():
    density = np.zeros((10, 10), np.float32)
    density[5, 5] = 1.0
    heat = colorize_density(density, (100, 100))
    assert heat.shape == (100, 100, 3)
    b, _, r = heat[55, 55]
    assert r > b  # dense = red (old code showed it blue)
    b, _, r = heat[0, 0]
    assert b > r


def test_colorize_density_all_zero():
    heat = colorize_density(np.zeros((4, 4), np.float32), (8, 8))
    assert heat.shape == (8, 8, 3)


def test_compose_draws_zone_without_touching_input():
    frame = np.zeros((100, 200, 3), np.uint8)
    zone = {"name": "A", "points": [(10, 10), (150, 10), (150, 90), (10, 90)]}
    result = {"boxes": [], "zones": [zone], "zone_counts": [("A", 3)]}
    out = compose(frame, result)
    assert frame.sum() == 0
    assert tuple(out[50, 10]) == zone_color(0)


def test_compose_gives_each_zone_its_own_color():
    frame = np.zeros((100, 200, 3), np.uint8)
    zones = [{"name": "A", "points": [(10, 10), (90, 10), (90, 90), (10, 90)]},
             {"name": "B", "points": [(110, 10), (190, 10), (190, 90), (110, 90)]}]
    out = compose(frame, {"boxes": [], "zones": zones, "zone_counts": [("A", 1), ("B", 2)]})
    assert tuple(out[50, 10]) == zone_color(0)
    assert tuple(out[50, 190]) == zone_color(1)
    assert zone_color(0) != zone_color(1)
