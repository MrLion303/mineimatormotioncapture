from __future__ import annotations

import math
import time
import traceback
from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Signal

from .camera import (
    BEST_MODEL_VARIANT,
    CameraWorker,
    _download_model,
    default_model_path,
)


def build_video_timeline(
    frame_count: int,
    source_fps: float,
) -> tuple[list[float], float]:
    """Build a timestamp for every source frame without changing playback speed."""
    count = int(frame_count)
    fps = float(source_fps)
    if count < 1:
        raise ValueError("El vídeo no contiene fotogramas.")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("No se pudo determinar una frecuencia FPS válida.")
    timestamps = [index / fps for index in range(count)]
    return timestamps, count / fps


def pose_timestamp_is_valid(
    captured_at: float,
    recording_start: float,
    recording_duration: float,
    source: str,
) -> bool:
    """Validate relative timestamps for both camera captures and imported clips."""
    elapsed = float(captured_at) - float(recording_start)
    if not math.isfinite(elapsed) or elapsed < 0.0:
        return False
    if source == "video":
        return True
    duration = float(recording_duration)
    return math.isfinite(duration) and duration >= 0.0 and elapsed <= duration + 0.002


class VideoWorker(QThread):
    """Process a video with the same MediaPipe Heavy model used by the camera."""

    pose_ready = Signal(object, float)
    processing_progress = Signal(int, int, float, float)
    pose_fps = Signal(float)
    status = Signal(str)
    failed = Signal(str)
    completed = Signal(int, float)

    def __init__(self, video_path: str | Path, parent=None):
        super().__init__(parent)
        self.video_path = Path(video_path)
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        capture = None
        landmarker = None
        try:
            if not self.video_path.is_file():
                raise FileNotFoundError(f"No se encontró el vídeo: {self.video_path}")

            self.status.emit("Abriendo el vídeo y leyendo sus propiedades...")
            capture = cv2.VideoCapture(str(self.video_path))
            if not capture.isOpened():
                raise RuntimeError(
                    "No se pudo abrir el vídeo. Comprueba que el archivo no esté "
                    "dañado y que su códec sea compatible."
                )

            frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            source_fps = float(capture.get(cv2.CAP_PROP_FPS))
            timestamps, duration = build_video_timeline(frame_count, source_fps)

            model_path = default_model_path(BEST_MODEL_VARIANT)
            if not model_path.is_file():
                _download_model(model_path, BEST_MODEL_VARIANT, self.status.emit)

            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            options = vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.55,
                min_pose_presence_confidence=0.55,
                min_tracking_confidence=0.60,
                output_segmentation_masks=False,
            )
            landmarker = vision.PoseLandmarker.create_from_options(options)
            self.status.emit(
                f"Procesando {frame_count} fotogramas a {source_fps:.2f} FPS "
                f"con el modelo Heavy de MediaPipe..."
            )

            started = time.monotonic()
            frames_processed = 0
            poses_detected = 0
            last_timestamp_ms = -1
            progress_interval = max(1, int(round(source_fps / 5.0)))

            for frame_index, captured_at in enumerate(timestamps):
                if not self._running:
                    return
                ok, frame = capture.read()
                if not ok:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = max(
                    last_timestamp_ms + 1,
                    int(round(captured_at * 1000.0)),
                )
                last_timestamp_ms = timestamp_ms
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                frames_processed += 1

                if result.pose_landmarks and result.pose_world_landmarks:
                    image_landmarks = CameraWorker._landmark_values(
                        result.pose_landmarks[0]
                    )
                    pose_data = {
                        "world": CameraWorker._landmark_values(
                            result.pose_world_landmarks[0]
                        ),
                        "image": image_landmarks,
                        "image_size": (int(frame.shape[1]), int(frame.shape[0])),
                    }
                    self.pose_ready.emit(pose_data, captured_at)
                    poses_detected += 1

                elapsed = max(0.001, time.monotonic() - started)
                if (
                    frames_processed % progress_interval == 0
                    or frames_processed == frame_count
                ):
                    rate = frames_processed / elapsed
                    remaining = max(0.0, (frame_count - frames_processed) / max(rate, 0.001))
                    self.processing_progress.emit(
                        frames_processed,
                        frame_count,
                        rate,
                        remaining,
                    )
                    self.pose_fps.emit(rate)

            if frames_processed == 0:
                raise RuntimeError("No se pudo leer ningún fotograma del vídeo.")

            actual_duration = frames_processed / source_fps
            elapsed = max(0.001, time.monotonic() - started)
            rate = frames_processed / elapsed
            self.processing_progress.emit(
                frames_processed,
                frames_processed,
                rate,
                0.0,
            )
            self.completed.emit(poses_detected, actual_duration)
            self.status.emit(
                f"Vídeo procesado: {frames_processed} fotogramas y "
                f"{poses_detected} poses detectadas en {actual_duration:.2f} segundos."
            )
        except Exception as exc:
            self.failed.emit(
                f"No se pudo procesar el vídeo con MediaPipe PoseLandmarker: {exc}\n\n"
                f"Traza técnica completa:\n{traceback.format_exc()}"
            )
        finally:
            if landmarker is not None:
                try:
                    landmarker.close()
                except Exception:
                    pass
            if capture is not None:
                capture.release()
