"""Start the desktop app:  python run_desktop.py"""
import sys
import traceback

from PyQt5 import QtWidgets


def show_error(exc_type, exc, tb):
    # PyQt5 aborts on uncaught exceptions in slots; show them instead.
    traceback.print_exception(exc_type, exc, tb)
    QtWidgets.QMessageBox.critical(None, "Error", f"{exc_type.__name__}: {exc}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    sys.excepthook = show_error
    try:
        from desktop.main_window import MainWindow
        window = MainWindow()
    except Exception as e:
        QtWidgets.QMessageBox.critical(None, "Startup error", f"{e}\n\nSee README > Troubleshooting.")
        sys.exit(1)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
