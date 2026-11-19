from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

import cv2

from .miobject import export_miobject, load_template
from .motionbert import create_motionbert_inferencer, process_motionbert_video
from .retarget import calculate_pose_controls
from .video import build_video_timeline


TRACKED_LANDMARKS = (11, 13, 15, 12, 14, 16, 23, 25, 27, 24, 26, 28)
CALIBRATION_PARTS = {"body", "head", "left_arm", "right_arm", "left_leg", "right_leg"}
MIN_CONFIDENCE = 0.35


def _landmark_confidences(pose: dict[str, Any]) -> list[float]:
    world = pose.get("world") or []
    confidences: list[float] = []
    for index in TRACKED_LANDMARKS:
        if index >= len(world) or len(world[index]) < 4:
            continue
        visibility = float(world[index][3])
        presence = float(world[index][4]) if len(world[index]) > 4 else visibility
        confidence = min(visibility, presence)
        if confidence >= MIN_CONFIDENCE:
            confidences.append(max(0.0, min(1.0, confidence)))
    return confidences


def _select_calibration_sample(
    samples: list[tuple[float, dict[str, Any]]],
    requested_time: float,
) -> tuple[float, dict[str, Any]]:
    """Choose the closest fully trackable neutral-reference frame."""
    candidates: list[tuple[float, float, float, dict[str, Any]]] = []
    for timestamp, pose in samples:
        controls = calculate_pose_controls(pose, reference_pose=pose)
        if not CALIBRATION_PARTS.issubset(controls):
            continue
        confidence_values = _landmark_confidences(pose)
        if len(confidence_values) < 10:
            continue
        mean_confidence = sum(confidence_values) / len(confidence_values)
        candidates.append(
            (abs(float(timestamp) - requested_time), -mean_confidence, float(timestamp), pose)
        )

    if not candidates:
        raise RuntimeError(
            "No se encontró una pose de calibración con torso, cabeza, ambos brazos "
            "y ambas piernas visibles. Revisa que el cuerpo completo aparezca en el vídeo."
        )

    _, _, timestamp, pose = min(candidates, key=lambda item: (item[0], item[1]))
    return timestamp, pose


def _inference_quality(
    samples: list[tuple[float, dict[str, Any]]],
    frame_count: int,
) -> dict[str, float | int]:
    confidence_values: list[float] = []
    complete_frames = 0
    for _, pose in samples:
        values = _landmark_confidences(pose)
        confidence_values.extend(values)
        if len(values) >= len(TRACKED_LANDMARKS):
            complete_frames += 1

    total_frames = max(1, int(frame_count))
    mean_confidence = (
        sum(confidence_values) / len(confidence_values)
        if confidence_values else 0.0
    )
    return {
        "pose_frames": len(samples),
        "pose_coverage_percent": round(100.0 * len(samples) / total_frames, 2),
        "complete_body_frames": complete_frames,
        "complete_body_coverage_percent": round(
            100.0 * complete_frames / total_frames, 2
        ),
        "mean_tracked_landmark_confidence": round(mean_confidence, 4),
        "tracked_landmark_observations": len(confidence_values),
    }



def _write_pose_diagnostics(
    output_path: Path,
    samples: list[tuple[float, dict[str, Any]]],
    calibration_pose: dict[str, Any],
) -> Path:
    """Persist raw 3D landmarks and retarget controls for regression review."""
    rows = []
    for timestamp, pose in samples:
        controls = calculate_pose_controls(pose, reference_pose=calibration_pose)
        rows.append({
            "timestamp": round(float(timestamp), 6),
            "world": pose.get("world") or [],
            "controls": controls,
        })
    path = output_path.with_suffix(".poses.json")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"engine": "RTMPose + MotionBERT", "samples": rows}, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def run_video_test(
    video_path: str | Path,
    template_path: str | Path,
    output_path: str | Path,
    calibration_time: float = 0.8,
    output_fps: int = 30,
    optimize_keyframes: bool = True,
    smooth_motion: bool = False,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    video_path = Path(video_path).expanduser().resolve()
    template_path = Path(template_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    report_path = (
        Path(report_path).expanduser().resolve()
        if report_path is not None
        else output_path.with_suffix(".diagnostico.json")
    )

    if not video_path.is_file():
        raise FileNotFoundError(f"No se encontró el vídeo: {video_path}")
    if not template_path.is_file():
        raise FileNotFoundError(f"No se encontró la plantilla: {template_path}")

    # Validate the Mine-imator rig before spending time on model inference.
    load_template(template_path)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {video_path}")
    try:
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        timestamps, duration = build_video_timeline(frame_count, source_fps)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()

    if not 0.0 <= calibration_time <= duration:
        raise ValueError(
            f"El instante de calibración ({calibration_time:.3f} s) queda fuera "
            f"del vídeo ({duration:.3f} s)."
        )

    print(
        f"Vídeo: {width}x{height} · {frame_count} fotogramas · "
        f"{source_fps:.3f} FPS · {duration:.3f} s",
        flush=True,
    )
    print("Inicializando RTMPose y MotionBERT...", flush=True)
    inferencer = create_motionbert_inferencer()

    samples: list[tuple[float, dict[str, Any]]] = []
    started = time.perf_counter()

    def on_status(message: str) -> None:
        print(message, flush=True)

    def on_pose(pose: dict[str, Any], timestamp: float) -> None:
        samples.append((float(timestamp), pose))

    processed = process_motionbert_video(
        video_path,
        timestamps,
        on_pose,
        on_status,
        inferencer=inferencer,
    )
    elapsed = max(0.001, time.perf_counter() - started)
    if processed < 1 or not samples:
        raise RuntimeError("RTMPose/MotionBERT no devolvió poses válidas para exportar.")

    calibration_timestamp, calibration_pose = _select_calibration_sample(
        samples, calibration_time
    )
    quality = _inference_quality(samples, frame_count)
    poses_path = _write_pose_diagnostics(output_path, samples, calibration_pose)

    print(
        f"Calibración neutral: {calibration_timestamp:.3f} s "
        f"(solicitada: {calibration_time:.3f} s)",
        flush=True,
    )
    export_result = export_miobject(
        template_path,
        output_path,
        samples,
        fps=output_fps,
        duration_seconds=duration,
        optimize_keyframes=optimize_keyframes,
        smooth_motion=smooth_motion,
        calibration_pose=calibration_pose,
    )

    warnings: list[str] = []
    if quality["pose_coverage_percent"] < 85.0:
        warnings.append(
            "La cobertura de poses es inferior al 85%; revisa los fotogramas sin detección."
        )
    if quality["complete_body_coverage_percent"] < 70.0:
        warnings.append(
            "La cobertura simultánea de las doce articulaciones principales es inferior al 70%."
        )
    if quality["mean_tracked_landmark_confidence"] < 0.60:
        warnings.append(
            "La confianza media de las articulaciones es baja; revisa oclusiones y encuadre."
        )

    report = {
        "video": {
            "path": str(video_path),
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "source_fps": source_fps,
            "duration_seconds": round(duration, 6),
        },
        "template": {
            "path": str(template_path),
            "format": load_template(template_path).get("format"),
        },
        "inference": {
            **quality,
            "processing_fps": round(processed / elapsed, 3),
            "processing_seconds": round(elapsed, 3),
            "engine": "RTMPose + MotionBERT",
            "device": "cpu",
        },
        "calibration": {
            "requested_timestamp": calibration_time,
            "selected_timestamp": round(calibration_timestamp, 6),
            "strategy": "nearest_complete_neutral_pose",
        },
        "pose_diagnostics_path": str(poses_path),
        "export": {
            **export_result,
            "duration_seconds": round(float(export_result["duration"]), 6),
        },
        "warnings": warnings,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_report.replace(report_path)

    print(
        f"Captura exportada: {output_path}\n"
        f"Diagnóstico: {report_path}\n"
        f"Cobertura: {quality['pose_coverage_percent']}% · "
        f"confianza media: {quality['mean_tracked_landmark_confidence']} · "
        f"keyframes: {export_result['total_keyframes']}",
        flush=True,
    )
    for warning in warnings:
        print(f"ADVERTENCIA: {warning}", flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Procesa un vídeo con RTMPose/MotionBERT y exporta un .miobject."
    )
    parser.add_argument("--test-video", required=True, help="Ruta del vídeo de entrada.")
    parser.add_argument("--template", required=True, help="Plantilla humana .miobject.")
    parser.add_argument("--output", required=True, help="Ruta del .miobject animado.")
    parser.add_argument(
        "--calibration-time",
        type=float,
        default=0.8,
        help="Segundo del vídeo que contiene la postura neutra (por defecto: 0.8).",
    )
    parser.add_argument("--fps", type=int, default=30, help="FPS de exportación (12-60).")
    parser.add_argument(
        "--smooth-motion",
        action="store_true",
        help="Aplica estabilización temporal al movimiento.",
    )
    parser.add_argument(
        "--no-optimize-keyframes",
        action="store_true",
        help="Conserva un keyframe por cada fotograma exportado.",
    )
    parser.add_argument("--report", help="Ruta opcional del informe JSON.")
    args = parser.parse_args(argv)

    try:
        report = run_video_test(
            args.test_video,
            args.template,
            args.output,
            calibration_time=args.calibration_time,
            output_fps=args.fps,
            optimize_keyframes=not args.no_optimize_keyframes,
            smooth_motion=args.smooth_motion,
            report_path=args.report,
        )
        return 0 if report["inference"]["pose_frames"] > 0 else 1
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
