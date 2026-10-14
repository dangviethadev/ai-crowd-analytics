import shutil

import pytest

from crowd.zones import load_zones
from helpers import MALL_IMAGE, VIDEO3, needs_demo_data

pytestmark = needs_demo_data


@pytest.fixture
def mall_copy(tmp_path):
    image = tmp_path / "mall.jpg"
    shutil.copy(MALL_IMAGE, image)
    return image


@pytest.fixture
def no_dialogs(monkeypatch):
    from PyQt5 import QtWidgets
    for name in ("information", "critical"):
        monkeypatch.setattr(QtWidgets.QMessageBox, name, lambda *a, **k: None)


def test_setup_loads_old_box_zone_as_polygon(qapp, mall_copy):
    import json
    from desktop.setup_window import SetupWindow
    old = {"drawList": [{"type": "box", "xy": {"x": 50, "y": 60, "width": 200, "height": 100}}]}
    mall_copy.with_suffix(".json").write_text(json.dumps(old), encoding="utf-8")
    win = SetupWindow()
    assert win.load_media(str(mall_copy))
    assert len(win.zones) == 1 and len(win.zones[0]["points"]) == 4
    assert win.tableWidget_DrawList.rowCount() == 1
    win.draw_zones(only=0)  # row click must not crash (old KeyError: 'config')


def test_draw_polygon_with_clicks_then_add_delete_save(qapp, mall_copy, no_dialogs):
    from desktop.setup_window import SetupWindow
    win = SetupWindow()
    assert not win.pushButton_SaveModel.isEnabled()  # nothing to save before a file is loaded
    assert win.load_media(str(mall_copy))
    view = win.view
    for x, y in [(-20, -20), (200, 10), (100, 150)]:
        view.add_point(x, y)
    view.close_polygon()
    assert win.pending_points == [(0, 0), (200, 10), (100, 150)]
    win.add_zone()
    for x, y in [(300, 200), (900, 200), (900, 900), (300, 900)]:
        view.add_point(x, y)
    view.close_polygon()
    win.add_zone()
    assert [z["name"] for z in win.zones] == ["Zone 1", "Zone 2"]
    win.delete_zone(0)
    win.save()
    assert load_zones(mall_copy) == [{"name": "Zone 2", "points": [(300, 200), (640, 200), (640, 480), (300, 480)]}]


def test_delete_button_removes_its_own_row(qapp, mall_copy):
    from desktop.setup_window import SetupWindow
    win = SetupWindow()
    win.show()
    win.load_media(str(mall_copy))
    for offset in (0, 300):
        for x, y in [(10 + offset, 10), (200 + offset, 10), (100 + offset, 150)]:
            win.view.add_point(x, y)
        win.view.close_polygon()
        win.add_zone()
    qapp.processEvents()
    table = win.tableWidget_DrawList
    assert len({id(table.cellWidget(r, 0)) for r in range(2)}) == 2
    table.cellWidget(1, 0).click()
    assert [z["name"] for z in win.zones] == ["Zone 1"]
    win.close()


def test_table_reuses_delete_buttons(qapp, mall_copy):
    from desktop.setup_window import SetupWindow
    win = SetupWindow()
    win.load_media(str(mall_copy))
    for x, y in [(10, 10), (200, 10), (100, 150)]:
        win.view.add_point(x, y)
    win.view.close_polygon()
    win.add_zone()
    button = win.tableWidget_DrawList.cellWidget(0, 0)
    win.refresh_table()
    assert win.tableWidget_DrawList.cellWidget(0, 0) is button  # no new widget per refresh


# Zones a user drew on Video3: Zone 1's bounding-box corner lies inside Zone 2.
OVERLAPPING = [{"name": "Zone 1", "points": [(15, 1069), (3, 796), (1663, 200), (1905, 200), (1908, 1063)]},
               {"name": "Zone 2", "points": [(6, 742), (1520, 209), (1390, 15), (0, 0), (3, 636)]}]


def test_zone_labels_sit_inside_their_zone_and_colors_differ(qapp):
    import cv2
    import numpy as np
    from PyQt5.QtWidgets import QGraphicsPolygonItem, QGraphicsTextItem
    from desktop.setup_window import SetupWindow
    win = SetupWindow()
    win.load_media(str(VIDEO3))
    win.zones = [dict(z) for z in OVERLAPPING]
    win.refresh_table()
    win.draw_zones()
    labels = {i.toPlainText(): i for i in win.scene.items() if isinstance(i, QGraphicsTextItem)}
    for zone in OVERLAPPING:
        c = labels[zone["name"]].sceneBoundingRect().center()
        assert cv2.pointPolygonTest(np.array(zone["points"], np.float32), (c.x(), c.y()), False) > 0
    colors = {i.pen().color().name() for i in win.scene.items() if isinstance(i, QGraphicsPolygonItem)}
    assert len(colors) == 2


def test_polygon_needs_three_points(qapp, mall_copy):
    from desktop.setup_window import SetupWindow
    win = SetupWindow()
    win.load_media(str(mall_copy))
    win.view.add_point(10, 10)
    win.view.add_point(50, 50)
    win.view.close_polygon()
    assert win.pending_points is None


def test_undo_and_cancel(qapp, mall_copy):
    from desktop.setup_window import SetupWindow
    win = SetupWindow()
    win.load_media(str(mall_copy))
    view = win.view
    for x, y in [(10, 10), (200, 10), (999, 999), (100, 150)]:
        view.add_point(x, y)
    view.undo_point()  # removes (100, 150)
    view.undo_point()  # removes (999, 999)
    view.add_point(100, 150)
    view.close_polygon()
    assert win.pending_points == [(10, 10), (200, 10), (100, 150)]
    view.add_point(5, 5)  # starting a new polygon discards the unsaved one
    assert win.pending_points is None
    view.cancel()
    assert view.points == []


def test_setup_rejects_unreadable_file(qapp, tmp_path, no_dialogs):
    from desktop.setup_window import SetupWindow
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"nope")
    assert not SetupWindow().load_media(str(bad))
