"""Zone setup: open an image or video, draw polygons, save them next to the file as JSON."""
from pathlib import Path

from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QBrush, QColor, QFont, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from crowd.config import ROOT
from crowd.media import read_first_frame, read_image
from crowd.render import zone_color
from crowd.zones import clamp_polygon, label_point, load_zones, save_zones
from desktop.qt_utils import to_qpixmap

UI_DIR = Path(__file__).resolve().parent / "ui"
DEMO_DIR = ROOT / "data" / "demo"
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp")
FILE_FILTER = "Images/Videos (*.png *.jpg *.jpeg *.bmp *.mp4 *.avi *.mov *.mkv *.wmv)"


class SetupWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi(str(UI_DIR / "Setup.ui"), self)
        self.view = self.label_Image
        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.image_loaded = False
        self.media_path = None
        self.image_size = (0, 0)
        self.zones = []
        self.pending_points = None

        self.pushButton_SelectImage.clicked.connect(self.open_file)
        self.pushButton_ConfirmDraw.clicked.connect(self.add_zone)
        self.pushButton_CheckDraw.clicked.connect(lambda: self.draw_zones())
        self.pushButton_SaveModel.clicked.connect(self.save)
        self.pushButton_ZoomIn.clicked.connect(self.zoom_in)
        self.pushButton_ZoomOut.clicked.connect(self.zoom_out)
        self.pushButton_FitView.clicked.connect(self.fit_view)
        self.pushButton_exit.clicked.connect(self.close)
        self.tableWidget_DrawList.cellClicked.connect(lambda row, _: self.draw_zones(only=row))
        self.tableWidget_DrawList.itemChanged.connect(self.rename_zone)
        self.set_editing_enabled(False)
        self.show_placeholder()

    def set_editing_enabled(self, enabled):
        for widget in (self.pushButton_ConfirmDraw, self.pushButton_CheckDraw,
                       self.pushButton_SaveModel, self.tableWidget_DrawList):
            widget.setEnabled(enabled)

    def show_pixmap(self, pixmap):
        self.view.clear_drawing()
        self.scene.clear()
        self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(QRectF(pixmap.rect()))
        self.fit_view()

    def show_placeholder(self):
        placeholder = UI_DIR / "images" / "backgroud-placeholder.png"
        if placeholder.exists():
            self.show_pixmap(QPixmap(str(placeholder)))

    def open_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open image or video", str(DEMO_DIR), FILE_FILTER)
        if path:
            self.load_media(path)

    def load_media(self, path):
        image = read_image(path) if path.lower().endswith(IMAGE_EXT) else read_first_frame(path)
        if image is None:
            QtWidgets.QMessageBox.critical(self, "Error", "Cannot read this file.")
            return False
        self.media_path = path
        self.image_size = (image.shape[1], image.shape[0])
        self.lineEdit_ImagePath.setText(path)
        self.show_pixmap(to_qpixmap(image))
        self.image_loaded = True
        self.pending_points = None
        self.zones = load_zones(path)
        self.refresh_table()
        self.set_editing_enabled(True)
        self.draw_zones()
        return True

    def on_polygon_drawn(self, points):
        self.pending_points = clamp_polygon(points, *self.image_size) if points else None

    def next_zone_name(self):
        used = {zone["name"] for zone in self.zones}
        i = 1
        while f"Zone {i}" in used:
            i += 1
        return f"Zone {i}"

    def add_zone(self):
        if self.pending_points is None:
            QtWidgets.QMessageBox.information(
                self, "Info", "Click on the image to add points, then double-click to close the zone.")
            return
        self.zones.append({"name": self.next_zone_name(), "points": self.pending_points})
        self.pending_points = None
        self.refresh_table()
        self.draw_zones()

    def delete_zone(self, index):
        del self.zones[index]
        self.refresh_table()
        self.draw_zones()

    def delete_clicked_zone(self):
        table = self.tableWidget_DrawList
        for row in range(table.rowCount()):
            if table.cellWidget(row, 0) is self.sender():
                self.delete_zone(row)
                return

    def rename_zone(self, item):
        if item.column() == 1 and item.text().strip():
            self.zones[item.row()]["name"] = item.text().strip()

    def refresh_table(self):
        table = self.tableWidget_DrawList
        table.blockSignals(True)
        table.setRowCount(len(self.zones))
        for row, zone in enumerate(self.zones):
            if table.cellWidget(row, 0) is None:  # buttons are reused; each finds its row when clicked
                button = QtWidgets.QPushButton("Delete")
                button.clicked.connect(self.delete_clicked_zone)
                table.setCellWidget(row, 0, button)
            name_item = QtWidgets.QTableWidgetItem(zone["name"])
            name_item.setForeground(QColor(*zone_color(row, bgr=False)))  # same color as on the image
            table.setItem(row, 1, name_item)
            points_item = QtWidgets.QTableWidgetItem(f"{len(zone['points'])} points")
            points_item.setFlags(points_item.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, 2, points_item)
        table.blockSignals(False)

    def draw_zones(self, only=None):
        """Draw saved zones; `only` is a row index to highlight a single zone."""
        self.view.clear_drawing()
        for item in self.scene.items():
            if not isinstance(item, QGraphicsPixmapItem):
                self.scene.removeItem(item)
        font = QFont("Segoe UI", max(12, self.image_size[0] // 50), QFont.Bold)
        for i, zone in enumerate(self.zones):
            if only is not None and i != only:
                continue
            color = QColor(*zone_color(i, bgr=False))
            pen = QPen(color, 3)
            pen.setCosmetic(True)
            fill = QColor(color)
            fill.setAlpha(50)
            self.scene.addPolygon(QPolygonF([QPointF(x, y) for x, y in zone["points"]]), pen, QBrush(fill))
            text = self.scene.addText(zone["name"], font)
            text.setDefaultTextColor(color)
            cx, cy = label_point(zone["points"])
            rect = text.boundingRect()
            text.setPos(cx - rect.width() / 2, cy - rect.height() / 2)

    def save(self):
        try:
            path = save_zones(self.media_path, self.zones)
        except OSError as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Cannot save zones: {e}")
            return None
        QtWidgets.QMessageBox.information(self, "Saved", f"Zones saved to:\n{path}")
        return path

    def fit_view(self):
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def zoom_in(self):
        self.view.scale(1.2, 1.2)

    def zoom_out(self):
        self.view.scale(1 / 1.2, 1 / 1.2)
