from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path


_stdio_log_handle = None
_stdio_log_path: Path | None = None


def _configure_stdio(log_directory: Path | None = None) -> Path | None:
    """Provide real text streams when PyInstaller starts a windowed process."""
    global _stdio_log_handle, _stdio_log_path

    if sys.stdout is not None and sys.stderr is not None:
        return None

    if _stdio_log_handle is None or _stdio_log_handle.closed:
        if log_directory is None:
            local_app_data = Path(
                os.environ.get("LOCALAPPDATA")
                or (Path.home() / "AppData" / "Local")
            )
            log_directory = (
                local_app_data
                / "NegativeStudios"
                / "MineimatorMotionCapture"
                / "logs"
            )

        try:
            log_directory.mkdir(parents=True, exist_ok=True)
            _stdio_log_path = log_directory / "runtime.log"
            _stdio_log_handle = _stdio_log_path.open(
                "a",
                encoding="utf-8",
                buffering=1,
            )
        except OSError:
            # A read-only profile must not prevent the app from starting.
            _stdio_log_handle = open(
                os.devnull,
                "a",
                encoding="utf-8",
                buffering=1,
            )
            _stdio_log_path = None

    if sys.stdout is None:
        sys.stdout = _stdio_log_handle
    if sys.stderr is None:
        sys.stderr = _stdio_log_handle

    return _stdio_log_path


def _run_mmpose_smoke_test() -> int:
    """Initialize both models inside the packaged executable without opening the UI."""
    try:
        from mocap_app.motionbert import create_motionbert_inferencer

        create_motionbert_inferencer()
        return 0
    except Exception:
        traceback.print_exc()
        log_path = Path.cwd() / "mmpose-smoke-test.log"
        try:
            log_path.write_text(traceback.format_exc(), encoding="utf-8")
        except OSError:
            pass
        return 1


def main() -> int:
    # PyInstaller --windowed may set sys.stdout and sys.stderr to None.
    # PyTorch's checkpoint downloader writes its progress to sys.stderr.
    _configure_stdio()

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
