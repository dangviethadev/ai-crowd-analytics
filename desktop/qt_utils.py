import cv2
from PyQt5 import QtGui


def to_qimage(img_bgr):
    """BGR numpy image -> QImage that owns its memory."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()


def to_qpixmap(img_bgr):
    return QtGui.QPixmap.fromImage(to_qimage(img_bgr))
