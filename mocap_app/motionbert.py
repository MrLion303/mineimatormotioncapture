from __future__ import annotations

from math import floor, isfinite
from pathlib import Path
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
            and abs(source_times[source_index + 1] - target_time)
            < abs(source_times[source_index] - target_time)
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
    batches = result.get("predictions") or []
    if not batches:
        return None
    first_batch = batches[0]
    if isinstance(first_batch, dict):
        return first_batch
    if isinstance(first_batch, list) and first_batch:
        return first_batch[0]
    return None


def process_motionbert_video(
    video_path: str | Path,
    timestamps: Iterable[float],
    pose_callback: Callable[[dict, float], None],
    status_callback: Callable[[str], None],
) -> int:
    """Run the official MMPose human3d pipeline over a complete captured clip."""
    from mmpose.apis import MMPoseInferencer

    captured_times = list(timestamps)
    if not captured_times:
        return 0

    status_callback("Cargando RTMPose-M, RTMDet-M y MotionBERT...")
    inferencer = MMPoseInferencer(pose3d="human3d", device="cpu")

    status_callback("Reconstruyendo el movimiento 3D con MotionBERT...")
    count = 0
    frame_index = 0
    for result in inferencer(str(video_path)):
        timestamp_index = min(frame_index, len(captured_times) - 1)
        frame_index += 1
        instance = _prediction_instance(result)
        if instance is None or instance.get("keypoints") is None:
            continue
        pose = map_h36m_keypoints_to_pose(
            instance["keypoints"],
            instance.get("keypoint_scores"),
        )
        pose_callback(pose, captured_times[timestamp_index])
        count += 1
        if frame_index % 30 == 0:
            status_callback(f"MotionBERT: {frame_index} fotogramas analizados...")

    return count
