from __future__ import annotations

import os
import queue
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage


MODEL_VARIANTS = {
    "heavy": {
        "label": "Heavy",
        "url": (
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
        ),
        "filename": "pose_landmarker_heavy.task",
    },
    "full": {
        "label": "Full",
        "url": (
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_full/float16/1/pose_landmarker_full.task"
        ),
        "filename": "pose_landmarker_full.task",
    },
    "lite": {
        "label": "Lite",
        "url": (
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        ),
        "filename": "pose_landmarker_lite.task",
    },
}

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


POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
)


def default_model_path(variant: str = "full") -> Path:
    model = MODEL_VARIANTS.get(variant, MODEL_VARIANTS["full"])
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    else:
        root = Path.home() / ".local" / "share"
    return (
        root
        / "NegativeStudios"
        / "MineimatorMotionCapture"
        / "models"
        / model["filename"]
    )


def _download_model(target: Path, variant: str, status_callback) -> None:
    model = MODEL_VARIANTS.get(variant, MODEL_VARIANTS["full"])
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".download")
    request = urllib.request.Request(
        model["url"],
        headers={"User-Agent": "MineimatorMotionCapture/0.2"},
    )
    status_callback(f"Descargando el modelo {model['label']} de seguimiento corporal...")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            with temporary.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
        if not temporary.exists() or temporary.stat().st_size < 500_000:
            raise RuntimeError("El archivo del modelo descargado está incompleto.")
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


class CameraWorker(QThread):
    pose_ready = Signal(object, float)
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
        model_variant: str = "full",
        parent=None,
        camera_backend: int | None = None,
        camera_name: str | None = None,
        preview_fps: int = 30,
    ):
        super().__init__(parent)
        self.camera_index = camera_index
        self.camera_backend = camera_backend
        self.camera_name = camera_name or f"Cámara {camera_index}"
        self.model_variant = model_variant if model_variant in MODEL_VARIANTS else "full"
        self._preview_fps = max(10, min(60, int(preview_fps)))
        self._preview_interval = 1.0 / self._preview_fps
        self._running = True

        # La cámara y MediaPipe trabajan por separado para que el análisis no
        # bloquee la lectura ni la presentación del vídeo.
        self._inference_queue = queue.Queue(maxsize=1)
        self._inference_ready = threading.Event()
        self._inference_error: str | None = None
        self._inference_thread: threading.Thread | None = None

        self._preview_lock = threading.Lock()
        self._latest_preview: tuple[int, QImage] | None = None
        self._preview_sequence = 0

        self._landmarks_lock = threading.Lock()
        self._latest_image_landmarks: list[list[float]] | None = None

    def stop(self) -> None:
        self._running = False

    def set_preview_fps(self, fps: int) -> None:
        self._preview_fps = max(10, min(60, int(fps)))
        self._preview_interval = 1.0 / self._preview_fps

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
        """Mantiene como máximo un frame pendiente, siempre el más reciente."""
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
            # Si el consumidor recuperó la cola justo ahora, no bloqueamos
            # nunca la captura para entregar una imagen antigua.
            pass

    @staticmethod
    def _landmark_values(points) -> list[list[float]]:
        return [
            [
                float(point.x),
                float(point.y),
                float(point.z),
                (
                    1.0
                    if getattr(point, "visibility", None) is None
                    else float(point.visibility)
                ),
                (
                    1.0
                    if getattr(point, "presence", None) is None
                    else float(point.presence)
                ),
            ]
            for point in points
        ]

    def _run_pose_inference(self, model_path: Path) -> None:
        landmarker = None
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            options = vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.45,
                min_pose_presence_confidence=0.45,
                min_tracking_confidence=0.45,
                output_segmentation_masks=False,
            )
            landmarker = vision.PoseLandmarker.create_from_options(options)
            self._inference_ready.set()

            last_timestamp_ms = -1
            processed_frames = 0
            stats_started_at = time.monotonic()

            while self._running:
                try:
                    frame, captured_at, timestamp_ms = self._inference_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = max(last_timestamp_ms + 1, timestamp_ms)
                last_timestamp_ms = timestamp_ms

                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                processed_frames += 1

                if result.pose_landmarks:
                    image_landmarks = self._landmark_values(result.pose_landmarks[0])
                    with self._landmarks_lock:
                        self._latest_image_landmarks = image_landmarks

                    if result.pose_world_landmarks:
                        pose_data = {
                            "world": self._landmark_values(result.pose_world_landmarks[0]),
                            "image": image_landmarks,
                        }
                        self.pose_ready.emit(pose_data, captured_at)

                now = time.monotonic()
                interval = now - stats_started_at
                if interval >= 1.0:
                    self.pose_fps.emit(processed_frames / interval)
                    processed_frames = 0
                    stats_started_at = now

        except Exception as exc:
            self._inference_error = str(exc)
            self.failed.emit(str(exc))
            self.stop()
        finally:
            self._inference_ready.set()
            if landmarker is not None:
                try:
                    landmarker.close()
                except Exception:
                    pass

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

            model_path = default_model_path(self.model_variant)
            if not model_path.exists():
                _download_model(model_path, self.model_variant, self.status.emit)

            if not self._running:
                return

            self.status.emit(
                f"Preparando seguimiento corporal ({MODEL_VARIANTS[self.model_variant]['label']}). "
                "La cámara usará la frecuencia disponible del dispositivo."
            )
            self._inference_thread = threading.Thread(
                target=self._run_pose_inference,
                args=(model_path,),
                name=f"PoseInference-{self.camera_index}",
                daemon=True,
            )
            self._inference_thread.start()

            while self._running and not self._inference_ready.wait(0.1):
                pass

            if not self._running or self._inference_error:
                return

            self.status.emit(f"{self.camera_name} lista. Colócate de cuerpo completo frente a ella.")
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

                # Se entrega una copia al hilo de IA sin esperar a que termine.
                inference_frame = frame.copy()
                self._enqueue_latest(
                    self._inference_queue,
                    (inference_frame, captured_at, timestamp_ms),
                )

                now = time.monotonic()
                interval = now - stats_started_at
                if interval >= 1.0:
                    # CAM reports the camera's actual delivered rate, not the
                    # user-selected preview refresh limit.
                    self.preview_fps.emit(camera_frames / interval)
                    camera_frames = 0
                    stats_started_at = now

                if captured_at - last_preview_published >= self._preview_interval:
                    with self._landmarks_lock:
                        latest_landmarks = self._latest_image_landmarks

                    preview_frame = frame.copy()
                    if latest_landmarks:
                        self._draw_pose(preview_frame, latest_landmarks)

                    mirrored = cv2.flip(preview_frame, 1)
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

    @staticmethod
    def _draw_pose(frame, landmarks) -> None:
        height, width = frame.shape[:2]
        points = []
        for landmark in landmarks:
            if isinstance(landmark, (list, tuple)):
                normalized_x = float(landmark[0])
                normalized_y = float(landmark[1])
                visibility = float(landmark[3]) if len(landmark) > 3 else 1.0
            else:
                normalized_x = float(landmark.x)
                normalized_y = float(landmark.y)
                visibility = float(getattr(landmark, "visibility", 1.0))

            x = int(max(0, min(width - 1, normalized_x * width)))
            y = int(max(0, min(height - 1, normalized_y * height)))
            points.append((x, y, visibility >= 0.25))

        for start, end in POSE_CONNECTIONS:
            if points[start][2] and points[end][2]:
                cv2.line(frame, points[start][:2], points[end][:2], (70, 210, 175), 3, cv2.LINE_AA)

        for x, y, visible in points:
            if visible:
                cv2.circle(frame, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 6, (70, 210, 175), 1, cv2.LINE_AA)
