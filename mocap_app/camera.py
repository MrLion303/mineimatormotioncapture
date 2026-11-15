from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage


MODEL_VARIANTS = {
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
    frame_ready = Signal(object)
    pose_ready = Signal(object, float)
    status = Signal(str)
    ready = Signal()
    failed = Signal(str)

    def __init__(self, camera_index: int, model_variant: str = "full", parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.model_variant = model_variant if model_variant in MODEL_VARIANTS else "full"
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        camera = None
        landmarker = None
        try:
            self.status.emit("Abriendo cámara...")
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            camera = cv2.VideoCapture(self.camera_index, backend)
            if not camera.isOpened():
                raise RuntimeError(
                    f"No se pudo abrir la cámara {self.camera_index}. "
                    "Prueba otro dispositivo o cierra las aplicaciones que la estén usando."
                )

            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            camera.set(cv2.CAP_PROP_FPS, 30)

            model_path = default_model_path(self.model_variant)
            if not model_path.exists():
                _download_model(model_path, self.model_variant, self.status.emit)

            self.status.emit(
                f"Preparando detector de cuerpo ({MODEL_VARIANTS[self.model_variant]['label']})..."
            )
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
            self.status.emit("Cámara lista. Colócate de cuerpo completo frente a ella.")
            self.ready.emit()

            last_timestamp_ms = -1
            failed_reads = 0

            while self._running:
                ok, frame = camera.read()
                if not ok:
                    failed_reads += 1
                    if failed_reads > 60:
                        raise RuntimeError("La cámara dejó de entregar imágenes.")
                    time.sleep(0.01)
                    continue
                failed_reads = 0
                captured_at = time.monotonic()

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp_ms = max(last_timestamp_ms + 1, int(captured_at * 1000))
                last_timestamp_ms = timestamp_ms
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.pose_landmarks:
                    image_landmarks = result.pose_landmarks[0]
                    self._draw_pose(frame, image_landmarks)

                    if result.pose_world_landmarks:
                        world_landmarks = result.pose_world_landmarks[0]
                        pose_data = {
                            "world": [
                                [
                                    float(point.x),
                                    float(point.y),
                                    float(point.z),
                                    float(getattr(point, "visibility", 1.0)),
                                    float(getattr(point, "presence", 1.0)),
                                ]
                                for point in world_landmarks
                            ],
                            "image": [
                                [
                                    float(point.x),
                                    float(point.y),
                                    float(point.z),
                                    float(getattr(point, "visibility", 1.0)),
                                    float(getattr(point, "presence", 1.0)),
                                ]
                                for point in image_landmarks
                            ],
                        }
                        self.pose_ready.emit(pose_data, captured_at)

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
                self.frame_ready.emit(image)

        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self._running = False
            if landmarker is not None:
                try:
                    landmarker.close()
                except Exception:
                    pass
            if camera is not None:
                camera.release()

    @staticmethod
    def _draw_pose(frame, landmarks) -> None:
        height, width = frame.shape[:2]
        points = []
        for landmark in landmarks:
            x = int(max(0, min(width - 1, landmark.x * width)))
            y = int(max(0, min(height - 1, landmark.y * height)))
            visible = float(getattr(landmark, "visibility", 1.0)) >= 0.25
            points.append((x, y, visible))

        for start, end in POSE_CONNECTIONS:
            if points[start][2] and points[end][2]:
                cv2.line(frame, points[start][:2], points[end][:2], (70, 210, 175), 3, cv2.LINE_AA)

        for x, y, visible in points:
            if visible:
                cv2.circle(frame, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 6, (70, 210, 175), 1, cv2.LINE_AA)
