from __future__ import annotations

from math import floor, isfinite
from pathlib import Path
import platform
from typing import Callable, Iterable

import numpy as np


# MMPose's human3d alias returns Human3.6M's 17-joint order.
# 0 pelvis, 1 right hip, 2 right knee, 3 right foot,
# 4 left hip, 5 left knee, 6 left foot, 7 spine, 8 thorax,
# 9 neck, 10 head, 11 left shoulder, 12 left elbow,
# 13 left wrist, 14 right shoulder, 15 right elbow, 16 right wrist.
H36M_TO_LANDMARKS = {
    11: 11,  # left shoulder
    12: 13,  # left elbow
    13: 15,  # left wrist
    14: 12,  # right shoulder
    15: 14,  # right elbow
    16: 16,  # right wrist
    4: 23,   # left hip
    5: 25,   # left knee
    6: 27,   # left foot / ankle
    1: 24,   # right hip
    2: 26,   # right knee
    3: 28,   # right foot / ankle
}

DEFAULT_VIDEO_FPS = 30.0

WINDOWS_DETECTOR_WEIGHTS = (
    "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/"
    "faster_rcnn_r50_fpn_1x_coco/"
    "faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
)


def _motionbert_inferencer_kwargs(platform_name: str | None = None) -> dict:
    """Build a known-compatible MMPose configuration for the current OS."""
    system = platform_name or platform.system()
    kwargs = {"pose3d": "human3d", "device": "cpu"}

    # MMPose 1.3.2's own Windows inferencer test replaces the default RTMDet
    # detector with Faster R-CNN because the RTMDet alias is not reliable there.
    if system.lower().startswith("win"):
        detector_config = (
            Path(__file__).resolve().parent
            / "assets"
            / "faster_rcnn_r50_fpn_coco.py"
        )
        if not detector_config.is_file():
            raise RuntimeError(
                "Falta la configuración Faster R-CNN necesaria para Windows: "
                f"{detector_config}"
            )
        kwargs.update(
            det_model=str(detector_config),
            det_weights=WINDOWS_DETECTOR_WEIGHTS,
            det_cat_ids=0,
        )
    return kwargs


def create_motionbert_inferencer():
    """Load the complete 2D + 3D inference stack before recording is enabled."""
    try:
        from mmpose.apis import MMPoseInferencer
    except Exception as exc:
        raise RuntimeError(
            "No se pudo importar MMPose. Comprueba las dependencias de "
            f"RTMPose, RTMDet y MotionBERT. Detalle: {exc}"
        ) from exc

    try:
        return MMPoseInferencer(**_motionbert_inferencer_kwargs())
    except Exception as exc:
        raise RuntimeError(
            "No se pudieron inicializar los modelos RTMPose/MotionBERT. "
            "Comprueba la conexión a Internet para descargar los pesos "
            f"la primera vez y revisa la instalación de OpenMMLab. Detalle: {exc}"
        ) from exc


def _reset_inferencer_session(inferencer) -> None:
    """Remove temporal tracking/sequence state between independent recordings."""
    backend = getattr(inferencer, "inferencer", None) or inferencer
    buffer = getattr(backend, "_buffer", None)
    if hasattr(buffer, "clear"):
        buffer.clear()
    if hasattr(backend, "_video_input"):
        backend._video_input = False
    if hasattr(inferencer, "_video_input"):
        inferencer._video_input = False



def resample_capture_timestamps(
    captured_timestamps: Iterable[float],
    fps: float = DEFAULT_VIDEO_FPS,
) -> tuple[list[int], list[float]]:
    """Select source frames on a uniform timeline for MotionBERT video input."""
    source_times = [float(value) for value in captured_timestamps]
    if not source_times:
        raise ValueError("La captura no contiene marcas de tiempo.")
    if not isfinite(fps) or fps <= 0:
        raise ValueError("La frecuencia de reconstrucción debe ser mayor que cero.")
    if any(not isfinite(value) for value in source_times):
        raise ValueError("La captura contiene marcas de tiempo no finitas.")
    if any(current <= previous for previous, current in zip(source_times, source_times[1:])):
        raise ValueError("Las marcas de tiempo de la captura deben ser crecientes.")

    interval = 1.0 / fps
    duration = source_times[-1] - source_times[0]
    sample_count = max(1, floor(duration * fps + 1e-9) + 1)
    target_times = [
        source_times[0] + index * interval
        for index in range(sample_count)
    ]

    source_indices: list[int] = []
    source_index = 0
    for target_time in target_times:
        while (
            source_index + 1 < len(source_times)
            and (
                abs(source_times[source_index + 1] - target_time) + 1e-9
                < abs(source_times[source_index] - target_time)
            )
        ):
            source_index += 1
        source_indices.append(source_index)

    return source_indices, target_times


def map_h36m_keypoints_to_pose(keypoints, keypoint_scores=None) -> dict:
    """Convert MMPose Human3.6M 3D joints to the landmark IDs used by retarget.py."""
    points = np.asarray(keypoints, dtype=np.float32)
    points = np.squeeze(points)
    if points.shape != (17, 3):
        raise ValueError(
            f"MotionBERT devolvió una pose 3D con forma {points.shape}; se esperaba (17, 3)."
        )
    if not np.isfinite(points).all():
        raise ValueError("MotionBERT devolvió coordenadas 3D no finitas.")

    if keypoint_scores is None:
        scores = np.ones(17, dtype=np.float32)
    else:
        scores = np.asarray(keypoint_scores, dtype=np.float32).squeeze()
        if scores.shape != (17,) or not np.isfinite(scores).all():
            raise ValueError("MotionBERT devolvió puntuaciones de articulación inválidas.")
        scores = np.clip(scores, 0.0, 1.0)

    # Pose3DInferencer reorders Human3.6M coordinates to [camera X, height,
    # depth] and negates X and depth. Convert that documented output back to
    # the retargeter's convention: X right, Y down, Z toward the camera (-Z).
    camera_points = np.column_stack(
        (-points[:, 0], -points[:, 2], -points[:, 1])
    ).astype(np.float32, copy=False)

    world = [[0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(33)]
    for source_index, target_index in H36M_TO_LANDMARKS.items():
        x, y, z = camera_points[source_index]
        confidence = float(scores[source_index])
        world[target_index] = [float(x), float(y), float(z), confidence, confidence]

    neck = camera_points[9]
    head = camera_points[10]
    left_shoulder = camera_points[11]
    right_shoulder = camera_points[14]
    shoulder_axis = right_shoulder - left_shoulder

    # Human3.6M has head and neck joints, but no nose or ears. Approximate
    # the face direction from the neck-to-head vector and derive a lateral
    # head axis from the shoulders. These are helper landmarks, not detections.
    nose = head + (head - neck) * 0.30
    ear_offset = shoulder_axis * 0.12
    face_confidence = float(min(scores[9], scores[10]))
    ear_confidence = float(min(scores[10], scores[11], scores[14]))
    world[0] = [float(v) for v in nose] + [face_confidence, face_confidence]
    world[7] = [float(v) for v in (head - ear_offset)] + [ear_confidence, ear_confidence]
    world[8] = [float(v) for v in (head + ear_offset)] + [ear_confidence, ear_confidence]

    return {"world": world, "image": []}


def _prediction_instance(result: dict):
    if not isinstance(result, dict):
        return None
    batches = result.get("predictions") or []
    if not isinstance(batches, (list, tuple)):
        return None

    for batch in batches:
        candidates = batch if isinstance(batch, (list, tuple)) else [batch]
        for candidate in candidates:
            if (
                isinstance(candidate, dict)
                and candidate.get("keypoints") is not None
            ):
                return candidate
    return None


def _coco_scores_to_h36m(scores) -> np.ndarray | None:
    """Convert RTMPose COCO confidence scores to MotionBERT's H36M joint order."""
    values = np.asarray(scores, dtype=np.float32).squeeze()
    if values.shape != (17,) or not np.isfinite(values).all():
        return None
    values = np.clip(values, 0.0, 1.0)

    # Match MMPose's own COCO -> Human3.6M keypoint conversion. Composite
    # joints use the weaker source confidence so occluded joints stay low.
    h36m = np.zeros(17, dtype=np.float32)
    h36m[0] = min(values[11], values[12])  # pelvis from both hips
    h36m[[1, 2, 3, 4, 5, 6]] = values[[12, 14, 16, 11, 13, 15]]
    h36m[8] = min(values[5], values[6])  # thorax from both shoulders
    h36m[7] = min(h36m[0], h36m[8])  # spine from pelvis and thorax
    h36m[9] = values[0]  # neck from nose
    h36m[10] = min(values[1], values[2])  # head from both eyes
    h36m[[11, 12, 13, 14, 15, 16]] = values[[5, 7, 9, 6, 8, 10]]
    return h36m


def _current_pose2d_h36m_scores(inferencer) -> np.ndarray | None:
    """Read the current frame's RTMPose scores from MMPose's 2D buffer."""
    backend = getattr(inferencer, "inferencer", None) or inferencer
    pose2d_model = getattr(backend, "pose2d_model", None)
    model = getattr(pose2d_model, "model", None)
    metadata = getattr(model, "dataset_meta", None)
    dataset_name = metadata.get("dataset_name") if hasattr(metadata, "get") else None
    if dataset_name not in {"coco", "posetrack18"}:
        return None

    buffer = getattr(backend, "_buffer", {})
    frames = buffer.get("pose_est_results_list") if hasattr(buffer, "get") else None
    if not frames or not frames[-1]:
        return None
    instances = frames[-1]
    sample = instances[0]
    pred_instances = getattr(sample, "pred_instances", None)
    scores = getattr(pred_instances, "keypoint_scores", None)
    if scores is None:
        return None
    return _coco_scores_to_h36m(scores)


def process_motionbert_video(
    video_path: str | Path,
    timestamps: Iterable[float],
    pose_callback: Callable[[dict, float], None],
    status_callback: Callable[[str], None],
    inferencer=None,
) -> int:
    """Run MMPose human3d over a clip and align each result to its source time."""
    captured_times = [float(value) for value in timestamps]
    if not captured_times:
        return 0
    if any(not isfinite(value) for value in captured_times):
        raise ValueError("La secuencia contiene marcas de tiempo no finitas.")
    if any(
        current <= previous
        for previous, current in zip(captured_times, captured_times[1:])
    ):
        raise ValueError("Las marcas de tiempo de la captura deben ser crecientes.")

    pose_inferencer = inferencer or create_motionbert_inferencer()
    _reset_inferencer_session(pose_inferencer)

    status_callback("Reconstruyendo el movimiento 3D con MotionBERT...")
    count = 0
    frame_index = 0
    try:
        for result in pose_inferencer(str(video_path)):
            if frame_index >= len(captured_times):
                raise RuntimeError(
                    "MMPose devolvió más fotogramas que los registrados en la captura. "
                    "Se detuvo para evitar desajustar la animación."
                )
            captured_at = captured_times[frame_index]
            frame_index += 1

            instance = _prediction_instance(result)
            if instance is None:
                continue
            # The 3D inferencer can emit placeholder scores of 1.0.
            # Prefer the confidence values from the RTMPose 2D detections.
            keypoint_scores = _current_pose2d_h36m_scores(pose_inferencer)
            if keypoint_scores is None:
                keypoint_scores = instance.get("keypoint_scores")
            pose = map_h36m_keypoints_to_pose(
                instance["keypoints"],
                keypoint_scores,
            )
            pose_callback(pose, captured_at)
            count += 1
            if frame_index % 30 == 0:
                status_callback(
                    f"MotionBERT: {frame_index} fotogramas analizados..."
                )
    finally:
        # MMPose 1.3.2's unified wrapper does not clear Pose3DInferencer's
        # sequence buffer between calls, so explicitly isolate each recording.
        _reset_inferencer_session(pose_inferencer)

    if frame_index == 0:
        raise RuntimeError(
            "MMPose no pudo leer ningún fotograma del vídeo temporal."
        )
    return count
