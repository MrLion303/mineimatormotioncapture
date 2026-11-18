from __future__ import annotations

import math
import time
import traceback
from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Signal


EXPORT_FPS = 30.0


def build_video_timeline(frame_count: int, source_fps: float) -> tuple[list[float], float]:
    """Create timestamps that preserve the source video's playback speed."""
    if int(frame_count) < 1:
        raise ValueError("El vídeo no contiene fotogramas o no informa su duración.")
    if not math.isfinite(float(source_fps)) or float(source_fps) <= 0.0:
        raise ValueError("No se pudo determinar la frecuencia (FPS) del vídeo.")

    fps = float(source_fps)
    timestamps = [index / fps for index in range(int(frame_count))]
    duration = int(frame_count) / fps
    return timestamps, duration


def pose_timestamp_is_valid(
    captured_at: float,
    recording_start: float,
    recording_duration: float,
    source: str,
) -> bool:
    """Validate pose timestamps from either a camera take or an imported clip."""
    elapsed = float(captured_at) - float(recording_start)
    if not math.isfinite(elapsed) or elapsed < 0.0:
        return False

    # Imported-video timestamps are relative to the beginning of the clip.
    # Its duration is not known to the UI until inference has finished, so do
    # not compare them against the still-zero camera recording duration.
    if source == "video":
        return True

    duration = float(recording_duration)
    return math.isfinite(duration) and duration >= 0.0 and elapsed <= duration + 0.002


class VideoWorker(QThread):
    """Run the same RTMPose + MotionBERT pipeline on a video file, without a camera."""

    pose_ready = Signal(object, float)
    processing_progress = Signal(int, int, float, float)
    pose_fps = Signal(float)
    status = Signal(str)
    failed = Signal(str)
    completed = Signal(int, float)

    def __init__(self, video_path: str | Path, parent=None):
        super().__init__(parent)
        self.video_path = Path(video_path)

    def run(self) -> None:
        capture = None
        try:
            if not self.video_path.is_file():
                raise FileNotFoundError(f"No se encontró el vídeo: {self.video_path}")

            self.status.emit("Comprobando el vídeo y leyendo su duración...")
            capture = cv2.VideoCapture(str(self.video_path))
            if not capture.isOpened():
                raise RuntimeError(
                    "No se pudo abrir el vídeo. Comprueba que el archivo no esté dañado "
                    "y que su códec sea compatible."
                )

            frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            source_fps = float(capture.get(cv2.CAP_PROP_FPS))
            timestamps, duration = build_video_timeline(frame_count, source_fps)
            capture.release()
            capture = None

            self.status.emit(
                f"Vídeo listo: {frame_count} fotogramas · {source_fps:.2f} FPS · "
                f"{duration:.2f} segundos. Inicializando RTMPose y MotionBERT..."
            )

            from .motionbert import create_motionbert_inferencer, process_motionbert_video

            inferencer = create_motionbert_inferencer()
            self.status.emit("Vídeo abierto. Reconstruyendo el movimiento 3D...")

            processing_started = time.monotonic()
            count = process_motionbert_video(
                self.video_path,
                timestamps,
                lambda pose, captured_at: self.pose_ready.emit(pose, captured_at),
                self.status.emit,
                inferencer=inferencer,
                progress_callback=(
                    lambda completed, total, fps, remaining: self.processing_progress.emit(
                        completed, total, fps, remaining
                    )
                ),
            )
            elapsed = max(0.001, time.monotonic() - processing_started)
            if count == 0:
                raise RuntimeError(
                    "MotionBERT no encontró poses válidas. Comprueba que el cuerpo "
                    "completo aparezca en el vídeo y que haya iluminación suficiente."
                )

            self.pose_fps.emit(count / elapsed)
            self.completed.emit(count, duration)
            self.status.emit(f"Vídeo procesado: {count} poses reconstruidas.")
        except Exception as exc:
            self.failed.emit(
                f"No se pudo procesar el vídeo con RTMPose/MotionBERT: {exc}\n\n"
                f"Traza técnica completa:\n{traceback.format_exc()}"
            )
        finally:
            if capture is not None:
                capture.release()
