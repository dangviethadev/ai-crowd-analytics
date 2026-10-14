import time

import pytest

from crowd.analyzer import DENSITY, DETECTION
from crowd.media import read_image
from helpers import MALL_IMAGE, needs_demo_data, needs_density_model, needs_detector_model

pytestmark = needs_demo_data


@pytest.fixture(scope="module")
def window(qapp):
    from desktop.main_window import MainWindow
    return MainWindow()


@needs_density_model
def test_density_result_shows_count_and_heatmap(window):
    window.show_result(window.analyzer.process(read_image(MALL_IMAGE), DENSITY))
    assert window.label_Result.text().startswith("Count:")
    assert window.label_View.pixmap() is not None and not window.label_View.pixmap().isNull()


@needs_detector_model
def test_detection_result_has_no_heatmap(window):
    window.show_result(window.analyzer.process(read_image(MALL_IMAGE), DETECTION))
    assert "Density mode" in window.label_View.text()


def _fake_result(n_zones):
    import numpy as np
    return {"overlay": np.zeros((48, 64, 3), np.uint8), "heatmap": None, "count": 386, "level": "High",
            "zone_counts": [(f"Zone {i + 1}", 10 * i) for i in range(n_zones)]}


def _text_fits(label):
    from PyQt5 import QtCore, QtGui
    flags = int(QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap)
    rect = QtGui.QFontMetrics(label.font()).boundingRect(QtCore.QRect(0, 0, label.width(), 2000), flags, label.text())
    return rect.height() <= label.height() and rect.width() <= label.width()


def test_result_label_fits_three_zones(window):
    window.show_result(_fake_result(3))
    assert all(f"Zone {i}" in window.label_Result.text() for i in (1, 2, 3))
    assert _text_fits(window.label_Result)  # measured with word wrap, so nothing is clipped


def test_result_label_summarizes_many_zones(window):
    window.show_result(_fake_result(7))
    text = window.label_Result.text()
    assert "Zone 2" in text and "Zone 3" not in text and "+5 more" in text  # 2 shown + 5 more = 7
    assert len(text.split("\n")) == 4
    assert _text_fits(window.label_Result)


def test_control_panel_fits_the_window(window):
    # Uses the fixed sizes from Main.ui, so it does not depend on fonts (tests run offscreen).
    # On Windows the menu bar and status bar take about 55 px of the 705 px window.
    tab = window.tabWidget.geometry()
    assert tab.y() + tab.height() <= window.minimumHeight() - 55
    panel = window.groupBox_7.maximumHeight() + window.groupBox_8.maximumHeight() + 40  # + margins and spacing
    assert panel <= tab.height()


def test_running_state_toggles_buttons(window):
    window.set_running(True)
    assert not window.pushButton_Video.isEnabled() and not window.comboBox_Mode.isEnabled()
    assert not window.pushButton_Zones.isEnabled()
    assert window.pushButton_Stop.isEnabled()
    window.stop()
    assert window.pushButton_Camera.isEnabled() and not window.pushButton_Stop.isEnabled()
    assert window.pushButton_Zones.isEnabled()


def test_zone_setup_is_easy_to_find(window, monkeypatch):
    from desktop import main_window
    opened = []
    monkeypatch.setattr(main_window.SetupWindow, "exec_", lambda self: opened.append(self))
    assert window.action_Config.text() == "Zone Setup"
    window.pushButton_Zones.click()
    window.action_Config.trigger()
    assert len(opened) == 2


def test_finished_signal_from_old_run_is_ignored(window):
    window.run_id = 5
    window.set_running(True)
    window.on_job_finished(4)
    assert window.pushButton_Stop.isEnabled()
    window.on_job_finished(5)
    assert not window.pushButton_Stop.isEnabled()


@needs_detector_model
def test_image_job_runs_in_worker_and_updates_ui(qapp, window):
    image = read_image(MALL_IMAGE)
    window.label_Result.setText("Ready")
    window.start_job(window.analyzer.process(image, DETECTION, []) for _ in range(1))
    window.worker.join(timeout=60)
    deadline = time.time() + 5
    while window.pushButton_Stop.isEnabled() and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    assert not window.pushButton_Stop.isEnabled()
    assert window.label_Result.text().startswith("Count:")
