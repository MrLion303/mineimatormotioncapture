from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from PySide6.QtCore import QCoreApplication

from .miobject import export_miobject
from .video import VideoWorker


def run_regression(video_path: Path, template_path: Path, output_path: Path) -> dict:
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir el video de prueba: {video_path}")
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()

    samples: list[tuple[float, dict]] = []
    worker_result: dict = {}
    errors: list[str] = []

    worker = VideoWorker(video_path)
    worker.pose_ready.connect(
        lambda pose, timestamp: samples.append((float(timestamp), pose))
    )
    worker.completed.connect(
        lambda count, duration: worker_result.update(
            pose_count=int(count),
            duration_seconds=float(duration),
        )
    )
    worker.failed.connect(errors.append)

    # Calling run() directly makes this a deterministic, headless regression test.
    # It executes the exact worker code used by the desktop application's video import.
    worker.run()

    if errors:
        raise RuntimeError(errors[-1])
    if not worker_result:
        raise RuntimeError("El proceso de video terminó sin emitir el resultado.")
    if not samples:
        raise RuntimeError(
            "MediaPipe Heavy no detectó poses en el video de referencia; "
            "no se puede generar una animación válida."
        )
    if worker_result["pose_count"] != len(samples):
        raise RuntimeError(
            "El recuento de poses del worker no coincide con las muestras recibidas."
        )

    export_info = export_miobject(
        template_path,
        output_path,
        samples,
        fps=30,
        duration_seconds=worker_result["duration_seconds"],
        optimize_keyframes=True,
        smooth_motion=True,
    )

    diagnostic = {
        "video": str(video_path),
        "template": str(template_path),
        "source_frame_count": frame_count,
        "source_fps": source_fps,
        "source_duration_seconds": (
            frame_count / source_fps if source_fps > 0 else None
        ),
        "processed_duration_seconds": worker_result["duration_seconds"],
        "pose_samples": len(samples),
        "pose_coverage_percent": (
            round(len(samples) / frame_count * 100.0, 2)
            if frame_count > 0 else None
        ),
        "export": export_info,
    }
    diagnostic_path = output_path.with_suffix(".diagnostico.json")
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_path.write_text(
        json.dumps(diagnostic, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return diagnostic


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prueba de regresión de importación de video con MediaPipe Heavy."
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    result = run_regression(args.video, args.template, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
