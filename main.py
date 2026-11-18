from __future__ import annotations

import sys
import traceback
from pathlib import Path


def _run_mmpose_smoke_test() -> int:
    """Initialize both models inside the packaged executable without opening the UI."""
    try:
        from mocap_app.motionbert import create_motionbert_inferencer

        create_motionbert_inferencer()
        return 0
    except Exception:
        log_path = Path.cwd() / "mmpose-smoke-test.log"
        try:
            log_path.write_text(traceback.format_exc(), encoding="utf-8")
        except OSError:
            pass
        return 1


def main() -> int:
    if "--smoke-test-mmpose" in sys.argv:
        return _run_mmpose_smoke_test()

    from PySide6.QtWidgets import QApplication

    from mocap_app.main_window import MainWindow

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
