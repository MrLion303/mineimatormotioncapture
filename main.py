from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from mocap_app.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setOrganizationName("NegativeStudios")
    app.setApplicationName("MineimatorMotionCapture")
    app.setApplicationDisplayName("Mine-imator Motion Capture")
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
