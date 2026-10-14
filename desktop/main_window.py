"""Main window: choose an image, video or camera; analysis runs in a worker thread."""
import threading
from pathlib import Path

import psutil
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from qt_thread_updater import get_updater

from crowd.analyzer import DENSITY, DETECTION, Analyzer
from crowd.config import ROOT, load_config
from crowd.media import open_video, read_image
from crowd.zones import load_zones
from desktop.qt_utils import to_qimage
from desktop.setup_window import SetupWindow

UI_DIR = Path(__file__).resolve().parent / "ui"
DEMO_DIR = ROOT / "data" / "demo"
IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp)"
VIDEO_FILTER = "Videos (*.mp4 *.avi *.mov *.mkv *.wmv)"
LEVEL_COLORS = {"Low": "rgb(0, 220, 0)", "Medium": "rgb(255, 210, 0)", "High": "rgb(255, 80, 80)"}


def result_text(result, max_zones=3):
    """Count on the first line, then one zone per line (the label fits 4 lines at 125% Windows scaling)."""
    zones = [f"{name}: {count}" for name, count in result["zone_counts"]]
    if len(zones) > max_zones:
        zones = zones[:max_zones - 1] + [f"+{len(zones) - max_zones + 1} more"]
    return "\n".join([f"Count: {result['count']} ({result['level']})"] + zones)


class MainWindow(QtWidgets.QMainWindow):
    error_signal = QtCore.pyqtSignal(str)

    def __init__(self, cfg=None):
        super().__init__()
        uic.loadUi(str(UI_DIR / "Main.ui"), self)
        self.cfg = cfg or load_config()
        self.analyzer = Analyzer(self.cfg)
        self.run_id = 0
        self.worker = None
        get_updater()  # create the updater in the GUI thread

        self.comboBox_Mode.addItems(["Density", "Detection"])
        self.label_Image.setScaledContents(False)
        self.label_View.setScaledContents(False)
        self.pushButton_Camera.clicked.connect(self.open_camera)
        self.pushButton_Video.clicked.connect(self.open_video)
        self.pushButton_Image.clicked.connect(self.open_image)
        self.pushButton_Stop.clicked.connect(self.stop)
        self.pushButton_Zones.clicked.connect(self.open_zone_setup)
        self.action_Config.triggered.connect(self.open_zone_setup)
        self.error_signal.connect(lambda text: QtWidgets.QMessageBox.critical(self, "Error", text))

        self.monitor_timer = QtCore.QTimer(self)
        self.monitor_timer.timeout.connect(self.update_pc_performance)
        self.monitor_timer.start(2000)
        self.set_running(False)
        self.label_Result.setText("Ready")
        if not self.analyzer.density_available:
            self.comboBox_Mode.setCurrentText("Detection")
            self.statusbar.showMessage("Density model not installed - Detection mode only")

    @property
    def mode(self):
        return DETECTION if self.comboBox_Mode.currentText() == "Detection" else DENSITY

    def open_camera(self):
        cap = open_video(self.cfg["camera_source"])
        if cap is None:
            self.error_signal.emit("Cannot open the camera. Check camera_source in config.yml.")
            return
        self.start_job(self.analyzer.stream(cap, self.mode, realtime=False), cap.release)

    def open_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open video", str(DEMO_DIR), VIDEO_FILTER)
        if not path:
            return
        cap = open_video(path)
        if cap is None:
            self.error_signal.emit("Cannot open this video.\nTip: avoid non-English characters in the file path.")
            return
        self.start_job(self.analyzer.stream(cap, self.mode, load_zones(path)), cap.release)

    def open_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open image", str(DEMO_DIR), IMAGE_FILTER)
        if not path:
            return
        image = read_image(path)
        if image is None:
            self.error_signal.emit("Cannot read this image.")
            return
        mode, zones = self.mode, load_zones(path)
        self.start_job(self.analyzer.process(image, mode, zones) for _ in range(1))

    def open_zone_setup(self):
        SetupWindow(self).exec_()

    def start_job(self, results, cleanup=None):
        """Consume a generator of results in a worker thread and show each one."""
        if self.worker is not None and self.worker.is_alive():
            self.worker.join(timeout=5)
        self.run_id += 1
        self.set_running(True)
        self.statusbar.showMessage("Running...")
        self.worker = threading.Thread(target=self._run_job, args=(results, self.run_id, cleanup), daemon=True)
        self.worker.start()

    def _run_job(self, results, run_id, cleanup):
        try:
            for result in results:
                if run_id != self.run_id:
                    break
                get_updater().call_latest(self.show_result, result)
        except Exception as e:
            self.error_signal.emit(f"Analysis failed: {e}")
        finally:
            results.close()
            if cleanup:
                cleanup()
            get_updater().call_latest(self.on_job_finished, run_id)

    def on_job_finished(self, run_id):
        if run_id == self.run_id:  # ignore jobs that were stopped or replaced
            self.set_running(False)
            self.statusbar.showMessage("Finished", 5000)

    def stop(self):
        self.run_id += 1
        self.set_running(False)
        self.statusbar.showMessage("Stopped", 5000)

    def set_running(self, running):
        for widget in (self.pushButton_Camera, self.pushButton_Video, self.pushButton_Image,
                       self.pushButton_Zones, self.comboBox_Mode, self.action_Config):
            widget.setEnabled(not running)
        self.pushButton_Stop.setEnabled(running)

    def show_result(self, result):
        self.set_label_image(self.label_Image, result["overlay"])
        if result["heatmap"] is not None:
            self.set_label_image(self.label_View, result["heatmap"])
        else:
            self.label_View.clear()
            self.label_View.setText("Heatmap is available\nin Density mode")
        self.label_Result.setText(result_text(result))
        self.label_Result.setStyleSheet(f"background-color: {LEVEL_COLORS[result['level']]};")

    @staticmethod
    def set_label_image(label, img_bgr):
        pixmap = QtGui.QPixmap.fromImage(to_qimage(img_bgr))
        label.setPixmap(pixmap.scaled(label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def update_pc_performance(self):
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage(ROOT.anchor or "/")
        self.progressBar_CPU.setValue(int(psutil.cpu_percent()))
        self.progressBar_RAM.setValue(int(mem.percent))
        self.progressBar_DISK.setValue(int(disk.percent))
        self.ram.setText(f"{mem.used / 1e9:.1f}/{mem.total / 1e9:.1f}")
        self.disk.setText(f"{disk.used / 1e9:.1f}/{disk.total / 1e9:.0f}")

    def closeEvent(self, event):
        reply = QtWidgets.QMessageBox.question(self, "Exit", "Do you want to exit?",
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                               QtWidgets.QMessageBox.Yes)
        if reply == QtWidgets.QMessageBox.Yes:
            self.run_id += 1
            event.accept()
        else:
            event.ignore()
