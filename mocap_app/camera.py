from __future__ import annotations

import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage


DEFAULT_PREVIEW_FPS = 30


@dataclass(frozen=True)
class CameraDevice:
    name: str
    index: int
    backend: int
    backend_name: str


def _camera_backend_name(backend: int) -> str:
    try:
        return str(cv2.videoio_registry.getBackendName(backend))
    except Exception:
        return "OpenCV"


def enumerate_available_cameras() -> list[CameraDevice]:
    """Return connected capture devices with their actual system names."""
    try:
        from cv2_enumerate_cameras import enumerate_cameras
    except Exception:
        enumerate_cameras = None

    if sys.platform == "win32":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    elif sys.platform == "darwin":
        backends = [cv2.CAP_AVFOUNDATION]
    else:
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

    if enumerate_cameras is not None:
        for preferred_backend in backends:
            try:
                camera_infos = enumerate_cameras(preferred_backend)
            except Exception:
                continue
            devices = []
            for info in camera_infos:
                backend = int(getattr(info, "backend", preferred_backend))
                devices.append(
                    CameraDevice(
                        name=str(getattr(info, "name", "") or f"Cámara {info.index}"),
                        index=int(info.index),
                        backend=backend,
                        backend_name=_camera_backend_name(backend),
                    )
                )
            if devices:
                return devices

    # Fallback for systems where camera enumeration is unavailable.
    for backend in backends:
        devices = []
        for index in range(10):
            capture = cv2.VideoCapture(index, backend)
            opened = capture.isOpened()
            capture.release()
            if opened:
                devices.append(
                    CameraDevice(
                        name=f"Cámara {index} (nombre no disponible)",
                        index=index,
                        backend=backend,
                        backend_name=_camera_backend_name(backend),
                    )
                )
        if devices:
            return devices
    return []


class CameraWorker(QThread):
    pose_ready = Signal(object, float)
    capture_drained = Signal()
    preview_fps = Signal(float)
    pose_fps = Signal(float)
    status = Signal(str)
    ready = Signal()
    failed = Signal(str)

    REQUESTED_WIDTH = 640
    REQUESTED_HEIGHT = 480

    def __init__(
        self,
        camera_index: int,
        parent=None,
        camera_backend: int | None = None,
        camera_name: str | None = None,
        preview_fps: int = DEFAULT_PREVIEW_FPS,
    ):
        super().__init__(parent)
        self.camera_index = camera_index
        self.camera_backend = camera_backend
        self.camera_name = camera_name or f"Cámara {camera_index}"
        self._preview_fps = max(10, min(60, int(preview_fps)))
        self._preview_interval = 1.0 / self._preview_fps
        self._running = True

        # REC frames are timestamped and processed as a complete sequence after recording.
        self._inference_queue = queue.Queue(maxsize=18_000)
        self._capture_mode_lock = threading.Lock()
        self._recording_capture = False
        self._draining_capture = False
        self._inference_busy = False
        self._inference_ready = threading.Event()
        self._inference_error: str | None = None
        self._inference_thread: threading.Thread | None = None
        self._motion_frames: list[tuple[bytes, float]] = []
        self._postprocessing_done = True

        self._preview_lock = threading.Lock()
        self._latest_preview: tuple[int, QImage] | None = None
        self._preview_sequence = 0

    def stop(self) -> None:
        self._running = False

    def begin_recording_capture(self) -> None:
        """Start a fresh timestamped inference queue for a new recording."""
        with self._capture_mode_lock:
            self._recording_capture = True
            self._draining_capture = False
            self._motion_frames.clear()
            self._postprocessing_done = False
            while True:
                try:
                    self._inference_queue.get_nowait()
                except queue.Empty:
                    break

    def finish_recording_capture(self) -> None:
        """Stop enqueueing new frames and process every frame still buffered."""
        with self._capture_mode_lock:
            self._recording_capture = False
            self._draining_capture = True
        self._notify_capture_drained()

    def _is_recording_capture(self) -> bool:
        """Return whether camera frames should be buffered for this recording."""
        with self._capture_mode_lock:
            return self._recording_capture and not self._draining_capture

    def _notify_capture_drained(self) -> None:
        should_emit = False
        with self._capture_mode_lock:
            if (
                self._draining_capture
                and not self._inference_busy
                and self._inference_queue.empty()
                and self._postprocessing_done
            ):
                self._draining_capture = False
                should_emit = True
        if should_emit:
            self.capture_drained.emit()

    def latest_preview_frame(self) -> tuple[int, QImage] | None:
        """Devuelve el frame más reciente sin crear una cola de señales Qt."""
        with self._preview_lock:
            return self._latest_preview

    def _publish_preview_frame(self, image: QImage) -> None:
        with self._preview_lock:
            self._preview_sequence += 1
            self._latest_preview = (self._preview_sequence, image)

    @staticmethod
    def _enqueue_latest(target_queue: queue.Queue, item: tuple) -> None:
        """Keep only the newest pending frame outside a recording."""
        while True:
            try:
                target_queue.get_nowait()
            except queue.Empty:
                break
        try:
            target_queue.put_nowait(item)
        except queue.Full:
            pass

    @staticmethod
    def _enqueue_recording(target_queue: queue.Queue, item: tuple) -> None:
        """Keep a bounded chronological buffer while preserving recent motion."""
        try:
            target_queue.put_nowait(item)
            return
        except queue.Full:
            pass
        try:
            target_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            target_queue.put_nowait(item)
        except queue.Full:
            pass

    def _run_pose_inference(self) -> None:
        try:
            from .motionbert import process_motionbert_video

            self._motionbert_processor = process_motionbert_video
            self._inference_ready.set()

            while self._running:
                with self._capture_mode_lock:
                    self._inference_busy = True
                try:
                    frame, captured_at, _timestamp_ms = self._inference_queue.get(timeout=0.1)
                except queue.Empty:
                    with self._capture_mode_lock:
                        self._inference_busy = False
                        should_process = (
                            self._draining_capture
                            and not self._postprocessing_done
                        )
                        if should_process:
                            self._inference_busy = True

                    if should_process:
                        try:
                            self._process_motionbert_capture()
                        except Exception as exc:
                            self._inference_error = str(exc)
                            self.failed.emit(
                                "No se pudo reconstruir la pose 3D con RTMPose/MotionBERT: "
                                f"{exc}"
                            )
                        finally:
                            with self._capture_mode_lock:
                                self._inference_busy = False
                                self._postprocessing_done = True
                            self._notify_capture_drained()
                        continue

                    self._notify_capture_drained()
                    continue

                with self._capture_mode_lock:
                    is_capture_frame = (
                        self._recording_capture or self._draining_capture
                    )
                    if is_capture_frame and isinstance(frame, bytes):
                        self._motion_frames.append((frame, captured_at))
                    self._inference_busy = False

                self._notify_capture_drained()

        except Exception as exc:
            self._inference_error = str(exc)
            self.failed.emit(
                "No se pudo iniciar RTMPose/MotionBERT. "
                "Comprueba que el entorno MMPose esté instalado: "
                f"{exc}"
            )
            self._inference_ready.set()
            self.capture_drained.emit()

    def _process_motionbert_capture(self) -> None:
        import tempfile

        from .motionbert import resample_capture_timestamps

        with self._capture_mode_lock:
            captured_frames = list(self._motion_frames)
            self._motion_frames.clear()

        if not captured_frames:
            self.status.emit("No se recibieron fotogramas para reconstruir.")
            return

        source_timestamps = [captured_at for _, captured_at in captured_frames]
        frame_indices, output_timestamps = resample_capture_timestamps(
            source_timestamps,
            fps=float(DEFAULT_PREVIEW_FPS),
        )
        self.status.emit(
            f"Normalizando {len(captured_frames)} fotogramas a "
            f"{len(frame_indices)} muestras temporales de 30 FPS..."
        )

        with tempfile.TemporaryDirectory(prefix="mineimator-motionbert-") as temp_dir:
            video_path = Path(temp_dir) / "captura.avi"
            first_encoded = captured_frames[frame_indices[0]][0]
            first_frame = cv2.imdecode(
                np.frombuffer(first_encoded, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if first_frame is None:
                raise RuntimeError("El primer fotograma de la captura está dañado.")

            height, width = first_frame.shape[:2]
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                float(DEFAULT_PREVIEW_FPS),
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError("No se pudo crear el video temporal para MotionBERT.")

            written_timestamps: list[float] = []
            last_valid_frame = first_frame
            try:
                for frame_index, captured_at in zip(frame_indices, output_timestamps):
                    encoded_frame = captured_frames[frame_index][0]
                    frame = cv2.imdecode(
                        np.frombuffer(encoded_frame, dtype=np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    if frame is None:
                        frame = last_valid_frame
                    else:
                        if frame.shape[1] != width or frame.shape[0] != height:
                            frame = cv2.resize(frame, (width, height))
                        last_valid_frame = frame

                    writer.write(frame)
                    written_timestamps.append(captured_at)
            finally:
                writer.release()

            if not written_timestamps:
                raise RuntimeError("No quedaron fotogramas válidos en el video temporal.")

            self.status.emit(
                "Procesando la grabación con RTMDet-M, RTMPose-M y MotionBERT. "
                "La reconstrucción puede tardar más que la grabación."
            )
            processing_started = time.monotonic()
            count = self._motionbert_processor(
                video_path,
                written_timestamps,
                lambda pose, captured_at: self.pose_ready.emit(pose, captured_at),
                self.status.emit,
            )
            processing_seconds = max(0.001, time.monotonic() - processing_started)
            if count:
                self.pose_fps.emit(count / processing_seconds)

        if count == 0:
            raise RuntimeError(
                "MotionBERT no encontró poses válidas. Comprueba la iluminación "
                "y que el cuerpo completo aparezca en el encuadre."
            )
        self.status.emit(f"Reconstrucción 3D terminada: {count} fotogramas.")

    def run(self) -> None:
        camera = None
        try:
            self.status.emit(f"Abriendo {self.camera_name}...")
            backend = self.camera_backend
            if backend is None:
                backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            camera = cv2.VideoCapture(self.camera_index, backend)
            if not camera.isOpened():
                raise RuntimeError(
                    f"No se pudo abrir {self.camera_name}. "
                    "Prueba otro dispositivo o cierra las aplicaciones que la estén usando."
                )

            # MJPG suele permitir modos de mayor frecuencia en webcams USB.
            # Si el controlador no lo admite, OpenCV conserva el formato disponible.
            if os.name == "nt":
                camera.set(
                    cv2.CAP_PROP_FOURCC,
                    cv2.VideoWriter_fourcc(*"MJPG"),
                )
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.REQUESTED_WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.REQUESTED_HEIGHT)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self._running:
                return

            self.status.emit(
                "Preparando el motor RTMPose-M + RTMDet-M + MotionBERT..."
            )
            self._inference_thread = threading.Thread(
                target=self._run_pose_inference,
                name=f"PoseInference-{self.camera_index}",
                daemon=True,
            )
            self._inference_thread.start()

            while self._running and not self._inference_ready.wait(0.1):
                pass

            if not self._running or self._inference_error:
                return

            self.status.emit(
                f"{self.camera_name} lista. Colócate de cuerpo completo frente a ella. "
                "La pose 3D se reconstruirá al terminar cada grabación."
            )
            self.ready.emit()

            failed_reads = 0
            camera_frames = 0
            stats_started_at = time.monotonic()
            last_timestamp_ms = -1
            last_preview_published = 0.0

            while self._running:
                ok, frame = camera.read()
                if not ok:
                    failed_reads += 1
                    if failed_reads > 60:
                        raise RuntimeError("La cámara dejó de entregar imágenes.")
                    time.sleep(0.005)
                    continue

                failed_reads = 0
                captured_at = time.monotonic()
                camera_frames += 1
                timestamp_ms = max(last_timestamp_ms + 1, int(captured_at * 1000))
                last_timestamp_ms = timestamp_ms

                if self._is_recording_capture():
                    source_height, source_width = frame.shape[:2]
                    resize_factor = min(
                        1.0,
                        self.REQUESTED_WIDTH / max(1, source_width),
                        self.REQUESTED_HEIGHT / max(1, source_height),
                    )
                    if resize_factor < 1.0:
                        inference_frame = cv2.resize(
                            frame,
                            (
                                max(1, int(round(source_width * resize_factor))),
                                max(1, int(round(source_height * resize_factor))),
                            ),
                            interpolation=cv2.INTER_AREA,
                        )
                    else:
                        inference_frame = frame
                    encoded_ok, encoded_frame = cv2.imencode(
                        ".jpg",
                        inference_frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 94],
                    )
                    if encoded_ok:
                        with self._capture_mode_lock:
                            if self._recording_capture and not self._draining_capture:
                                self._enqueue_recording(
                                    self._inference_queue,
                                    (encoded_frame.tobytes(), captured_at, timestamp_ms),
                                )
                now = time.monotonic()
                interval = now - stats_started_at
                if interval >= 1.0:
                    # CAM reports the camera's actual delivered rate.
                    self.preview_fps.emit(camera_frames / interval)
                    camera_frames = 0
                    stats_started_at = now

                if captured_at - last_preview_published >= self._preview_interval:
                    mirrored = cv2.flip(frame, 1)
                    rgb_preview = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)
                    height, width, _ = rgb_preview.shape
                    image = QImage(
                        rgb_preview.data,
                        width,
                        height,
                        int(rgb_preview.strides[0]),
                        QImage.Format.Format_RGB888,
                    ).copy()
                    self._publish_preview_frame(image)
                    last_preview_published = captured_at

        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self._running = False
            if camera is not None:
                camera.release()
            if self._inference_thread is not None:
                self._inference_thread.join()
