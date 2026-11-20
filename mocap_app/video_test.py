from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from PySide6.QtCore import QCoreApplication

from .miobject import export_miobject
from .retarget import calculate_pose_controls, calculate_pose_sequence
from .video import VideoWorker


def run_regression(
    video_path: Path,
    template_path: Path,
    output_path: Path,
    fps: int = 24,
) -> dict:
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
        fps=fps,
        duration_seconds=worker_result["duration_seconds"],
        optimize_keyframes=True,
        smooth_motion=True,
    )

    exported = json.loads(output_path.read_text(encoding="utf-8"))
    body_timeline = next(
        timeline for timeline in exported["timelines"]
        if timeline.get("type") == "bodypart"
        and timeline.get("model_part_name") == "body"
    )
    body_keyframes = body_timeline.get("keyframes", {})
    body_fields = {
        field: [
            float(values[field])
            for values in body_keyframes.values()
            if field in values
        ]
        for field in ("BEND_ANGLE_X", "BEND_ANGLE_Y", "BEND_ANGLE_Z")
    }
    body_ranges = {
        field: {"min": min(values), "max": max(values)}
        for field, values in body_fields.items()
        if values
    }

    reference_pose = next(
        (
            pose for _, pose in samples
            if "body" in calculate_pose_controls(pose)
        ),
        samples[0][1],
    )
    pose_frames = []
    channel_values: dict[str, dict[str, list[float]]] = {}
    sequence_controls = calculate_pose_sequence(samples, reference_pose=reference_pose)
    for (timestamp, pose), (_, controls) in zip(samples, sequence_controls):
        pose_frames.append({
            "timestamp": round(float(timestamp), 6),
            "world": pose.get("world", []),
            "image": pose.get("image", []),
            "image_size": pose.get("image_size"),
            "controls": controls,
        })
        for part, fields in controls.items():
            part_values = channel_values.setdefault(part, {})
            for field, value in fields.items():
                part_values.setdefault(field, []).append(float(value))

    channel_ranges = {
        part: {
            field: {"min": min(values), "max": max(values)}
            for field, values in fields.items() if values
        }
        for part, fields in channel_values.items()
    }

    # A limb's hinge can flex, but the monocular pose landmarks do not provide
    # observable axial twist. Reject runaway Euler turns instead of silently
    # exporting them into a Mine-imator project.
    limb_rotation_peaks = {}
    for part in ("left_arm", "right_arm", "left_leg", "right_leg"):
        fields = channel_values.get(part, {})
        values = [
            abs(value)
            for channel in ("ROT_X", "ROT_Y", "ROT_Z")
            for value in fields.get(channel, [])
        ]
        peak = max(values, default=0.0)
        limb_rotation_peaks[part] = round(peak, 4)
        if peak > 180.0:
            raise RuntimeError(
                f"El retargeting produjo un giro axial no fiable en {part}: "
                f"{peak:.1f} grados. Se cancela la regresión para evitar "
                "keyframes de vueltas artificiales."
            )
    poses_path = output_path.with_suffix(".poses.json")
    poses_path.parent.mkdir(parents=True, exist_ok=True)
    poses_path.write_text(
        json.dumps({
            "video": str(video_path),
            "source_fps": source_fps,
            "frames": pose_frames,
        }, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    diagnostic = {
        "video": str(video_path),
        "template": str(template_path),
        "pose_diagnostics": poses_path.name,
        "first_frame_controls": pose_frames[0]["controls"],
        "raw_channel_ranges": channel_ranges,
        "limb_rotation_peak_degrees": limb_rotation_peaks,
        "torso_yaw_channel": "BEND_ANGLE_Z",
        "project_fps": int(fps),
        "mineimator_playback_duration_seconds": (
            export_info["final_frame"] / float(fps)
        ),
        "body_channel_ranges": body_ranges,
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
        "ik_enabled_parts": export_info.get("ik_enabled_parts", []),
        "ik_helper_timelines": export_info.get("ik_helper_timelines", 0),
        "ik_helper_keyframes": export_info.get("ik_helper_keyframes", {}),
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
    parser.add_argument(
        "--fps",
        type=int,
        choices=(24, 30, 60),
        default=24,
        help="FPS del proyecto de Mine-imator (predeterminado: 24).",
    )
    args = parser.parse_args()

    result = run_regression(
        args.video, args.template, args.output, fps=args.fps
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
