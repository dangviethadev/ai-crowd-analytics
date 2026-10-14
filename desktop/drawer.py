from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainterPath, QPen
from PyQt5.QtWidgets import QGraphicsView, QMenu


class drawer(QGraphicsView):  # lower-case name must match the custom widget in Setup.ui
    """Polygon drawing: click to add points, double-click to close,
    Backspace to remove the last point, Esc to cancel. Points are in image coordinates."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup = parent
        self.points = []
        self.path_item = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    def contextMenuEvent(self, event):
        menu = QMenu()
        menu.addAction("Zoom in", self.setup.zoom_in)
        menu.addAction("Zoom out", self.setup.zoom_out)
        menu.addAction("Fit view", self.setup.fit_view)
        menu.exec_(event.globalPos())

    # --- drawing actions (also used by tests)
    def add_point(self, x, y):
        if not self.points:
            self.clear_drawing()
            self.setup.on_polygon_drawn(None)  # a new polygon replaces one that was not added
        self.points.append((x, y))
        self.update_preview()

    def close_polygon(self):
        if len(self.points) >= 3:
            self.update_preview(closed=True)
            self.setup.on_polygon_drawn([(int(round(x)), int(round(y))) for x, y in self.points])
            self.points = []

    def undo_point(self):
        if self.points:
            self.points.pop()
            self.update_preview()

    def cancel(self):
        self.clear_drawing()
        self.setup.on_polygon_drawn(None)

    def clear_drawing(self):
        if self.path_item is not None and self.path_item.scene() is not None:
            self.scene().removeItem(self.path_item)
        self.path_item = None
        self.points = []

    def update_preview(self, cursor=None, closed=False):
        points = self.points + ([cursor] if cursor else [])
        path = QPainterPath()
        if points:
            path.moveTo(*points[0])
            for p in points[1:]:
                path.lineTo(*p)
            if closed:
                path.closeSubpath()
        if self.path_item is None or self.path_item.scene() is None:
            pen = QPen(Qt.red, 2, Qt.DashLine)
            pen.setCosmetic(True)
            self.path_item = self.scene().addPath(path, pen)
        else:
            self.path_item.setPath(path)

    # --- Qt events
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.setup.image_loaded:
            self.setFocus()
            p = self.mapToScene(event.pos())
            self.add_point(p.x(), p.y())
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        # The first click of a double-click already added the last point.
        if event.button() == Qt.LeftButton and self.setup.image_loaded:
            self.close_polygon()
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        if self.points:
            p = self.mapToScene(event.pos())
            self.update_preview(cursor=(p.x(), p.y()))
        super().mouseMoveEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and (self.points or self.path_item is not None):
            self.cancel()  # handled here so Esc does not close the dialog
        elif event.key() == Qt.Key_Backspace:
            self.undo_point()
        else:
            super().keyPressEvent(event)
