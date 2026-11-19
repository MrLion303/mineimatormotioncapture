from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path
from typing import Any, Iterable

from .retarget import WORLD_REFERENCE_FRAME, _torso_frame, calculate_pose_controls


REQUIRED_PARTS = {"body", "head", "left_arm", "right_arm", "left_leg", "right_leg"}
PART_FIELDS = {
    "body": ("ROT_X", "ROT_Y", "ROT_Z"),
    "head": ("ROT_X", "ROT_Z"),
    "left_arm": ("ROT_X", "ROT_Y", "ROT_Z", "BEND_ANGLE_X"),
    "right_arm": ("ROT_X", "ROT_Y", "ROT_Z", "BEND_ANGLE_X"),
    "left_leg": ("ROT_X", "ROT_Y", "ROT_Z", "BEND_ANGLE_X"),
    "right_leg": ("ROT_X", "ROT_Y", "ROT_Z", "BEND_ANGLE_X"),
}

KEYFRAME_ANGLE_THRESHOLD = 0.8
MAX_KEYFRAME_INTERVAL_SECONDS = 0.15

LEG_MOTION_GAINS = {
    "ROT_X": 1.0,
    "ROT_Z": 1.0,
    "BEND_ANGLE_X": 1.0,
}
LEG_MOTION_DEADZONES = {
    "ROT_X": 1.0,
    "ROT_Z": 1.0,
    "BEND_ANGLE_X": 1.5,
}


def _shape_motion_delta(part: str, field: str, value: float) -> float:
    """Reduce small lower-body overreaction without suppressing large gestures."""
    if part not in {"left_leg", "right_leg"}:
        return float(value)
    magnitude = abs(float(value))
    deadzone = LEG_MOTION_DEADZONES.get(field, 0.0)
    if magnitude <= deadzone:
        return 0.0
    gain = LEG_MOTION_GAINS.get(field, 1.0)
    return (magnitude - deadzone) * gain * (1.0 if value >= 0 else -1.0)


class MiobjectError(ValueError):
    pass


def load_template(path: str | Path) -> dict[str, Any]:
    template_path = Path(path)
    try:
        with template_path.open("r", encoding="utf-8-sig") as source:
            data = json.load(source)
    except (OSError, json.JSONDecodeError) as exc:
        raise MiobjectError(f"No se pudo leer el archivo .miobject: {exc}") from exc
    # A Mine-imator project is also a valid source rig. Strip editor-only
    # project/render/background settings before emitting a .miobject asset.
    if isinstance(data, dict):
        data = dict(data)
        for editor_section in ("project", "render", "background"):
            data.pop(editor_section, None)
    validate_template(data)
    return data


def validate_template(data: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise MiobjectError("El archivo no contiene un objeto JSON válido.")
    if not isinstance(data.get("timelines"), list):
        raise MiobjectError("El archivo no contiene la lista de pistas 'timelines'.")
    templates = data.get("templates", [])
    is_human = any(
        isinstance(item, dict)
        and isinstance(item.get("model"), dict)
        and str(item["model"].get("name", "")).lower() == "human"
        for item in templates
    )
    if not is_human:
        raise MiobjectError("La plantilla debe contener un personaje humano de Mine-imator.")
    found = {
        item.get("model_part_name")
        for item in data["timelines"]
        if isinstance(item, dict) and item.get("type") == "bodypart"
    }
    missing = REQUIRED_PARTS - found
    if missing:
        names = ", ".join(sorted(missing))
        raise MiobjectError(f"Faltan pistas corporales necesarias en la plantilla: {names}")


def _first_keyframe_data(timeline: dict[str, Any]) -> dict[str, Any]:
    keyframes = timeline.get("keyframes")
    if not isinstance(keyframes, dict) or not keyframes:
        return {}
    try:
        first_frame = min(keyframes, key=lambda frame: float(frame))
    except (TypeError, ValueError):
        first_frame = next(iter(keyframes))
    values = keyframes.get(first_frame, {})
    return dict(values) if isinstance(values, dict) else {}


def _template_static_part_fields(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Keep rig-specific fields that the motion retargeter does not animate."""
    static_fields: dict[str, dict[str, Any]] = {}
    for timeline in data.get("timelines", []):
        if not isinstance(timeline, dict) or timeline.get("type") != "bodypart":
            continue
        part = timeline.get("model_part_name")
        if not isinstance(part, str):
            continue
        dynamic_fields = set(PART_FIELDS.get(part, ()))
        values: dict[str, Any] = {}
        defaults = timeline.get("default_values")
        if isinstance(defaults, dict):
            values.update(defaults)
        values.update(_first_keyframe_data(timeline))
        preserved = {
            str(key): value
            for key, value in values.items()
            if key not in dynamic_fields and key != "TRANSITION"
        }
        if preserved:
            static_fields[part] = preserved
    return static_fields


def _static_first_keyframe(timeline: dict[str, Any]) -> None:
    values = _first_keyframe_data(timeline)
    values.pop("TRANSITION", None)
    timeline["keyframes"] = {"0": values} if values else {}


def _unwrap_degrees(values: list[float]) -> list[float]:
    if not values:
        return values
    output = [values[0]]
    for value in values[1:]:
        adjusted = value
        while adjusted - output[-1] > 180.0:
            adjusted -= 360.0
        while adjusted - output[-1] < -180.0:
            adjusted += 360.0
        output.append(adjusted)
    return output


def _angle_difference(value: float, reference: float) -> float:
    """Return the shortest signed angular offset from the calibrated stance."""
    return (float(value) - float(reference) + 180.0) % 360.0 - 180.0


def _interpolate(times: list[float], values: list[float], target: float) -> float:
    if len(times) == 1 or target <= times[0]:
        return values[0]
    if target >= times[-1]:
        return values[-1]
    right = bisect_right(times, target)
    left = right - 1
    span = times[right] - times[left]
    if span <= 1e-9:
        return values[right]
    amount = (target - times[left]) / span
    return values[left] + (values[right] - values[left]) * amount


def _smooth(values: list[float], enabled: bool, fps: int = 30) -> list[float]:
    if not enabled or len(values) < 3:
        return values

    # A 3-frame median rejects isolated landmark jumps. A One-Euro-style filter
    # then smooths slow tracking noise while raising its cutoff during fast
    # gestures to avoid the sluggish response of a fixed low-pass filter.
    from statistics import median

    robust = list(values)
    for index in range(1, len(values) - 1):
        robust[index] = median(values[index - 1:index + 2])

    dt = 1.0 / max(1, int(fps))

    def alpha_for(cutoff: float) -> float:
        tau = 1.0 / (2.0 * 3.141592653589793 * cutoff)
        return 1.0 / (1.0 + tau / dt)

    filtered = [robust[0]]
    last_raw = robust[0]
    derivative = 0.0
    derivative_alpha = alpha_for(1.0)
    for value in robust[1:]:
        raw_derivative = (value - last_raw) / dt
        derivative += derivative_alpha * (raw_derivative - derivative)
        cutoff = 1.8 + 0.025 * abs(derivative)
        alpha = alpha_for(cutoff)
        filtered.append(filtered[-1] + alpha * (value - filtered[-1]))
        last_raw = value
    return filtered


def _select_keyframes(
    frame_values: list[dict[str, float]],
    fields: tuple[str, ...],
    fps: int,
    optimize: bool,
) -> list[int]:
    if len(frame_values) <= 2:
        return list(range(len(frame_values)))
    if not optimize:
        return list(range(len(frame_values)))

    max_gap = max(2, round(fps * MAX_KEYFRAME_INTERVAL_SECONDS))
    selected = [0]
    previous = 0
    for index in range(1, len(frame_values) - 1):
        difference = max(
            abs(frame_values[index][field] - frame_values[previous][field])
            for field in fields
        )
        if difference >= KEYFRAME_ANGLE_THRESHOLD or index - previous >= max_gap:
            selected.append(index)
            previous = index
    if selected[-1] != len(frame_values) - 1:
        selected.append(len(frame_values) - 1)
    return selected


def export_miobject(
    template_path: str | Path,
    output_path: str | Path,
    samples: Iterable[tuple[float, dict[str, Any]]],
    fps: int = 30,
    duration_seconds: float | None = None,
    optimize_keyframes: bool = True,
    smooth_motion: bool = False,
    calibration_pose: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not 12 <= int(fps) <= 60:
        raise MiobjectError("Los FPS de salida deben estar entre 12 y 60.")

    data = load_template(template_path)
    samples = list(samples)
    if not samples:
        raise MiobjectError("No hay muestras de pose para exportar.")

    # Use an explicit neutral calibration pose when supplied (for example,
    # a frame selected from a neutral window in an imported video). Otherwise
    # retain the original behavior and calibrate against the first reliable pose.
    reference_pose = calibration_pose
    if reference_pose is None:
        reference_pose = next(
            (
                pose for _, pose in samples
                if "body" in calculate_pose_controls(pose)
            ),
            samples[0][1],
        )
    if "body" not in calculate_pose_controls(
        reference_pose, reference_pose=reference_pose
    ):
        raise MiobjectError(
            "La pose de calibración no contiene hombros y caderas visibles."
        )
    static_part_fields = _template_static_part_fields(data)
    converted: list[tuple[float, dict[str, dict[str, float]]]] = []
    previous_frame = _torso_frame(reference_pose) or WORLD_REFERENCE_FRAME
    for timestamp, pose in samples:
        controls = calculate_pose_controls(
            pose,
            reference_pose=reference_pose,
            previous_frame=previous_frame,
        )
        current_frame = _torso_frame(pose, fallback_frame=previous_frame)
        if current_frame is not None:
            previous_frame = current_frame
        if controls:
            converted.append((max(0.0, float(timestamp)), controls))
    if not converted:
        raise MiobjectError(
            "No hay poses válidas para exportar. Comprueba que se vean los hombros y las caderas."
        )

    converted.sort(key=lambda item: item[0])
    deduplicated: dict[float, dict[str, dict[str, float]]] = {}
    for timestamp, controls in converted:
        deduplicated[timestamp] = controls
    converted = sorted(deduplicated.items(), key=lambda item: item[0])

    # Calibrate each channel against the selected neutral pose. If a body
    # part is occluded in that frame, fall back to its first reliable sample.
    reference: dict[str, dict[str, float]] = {}
    calibration_controls = calculate_pose_controls(
        reference_pose, reference_pose=reference_pose
    )
    for part, fields in calibration_controls.items():
        reference[part] = {
            field: float(value) for field, value in fields.items()
        }
    for _, controls in converted:
        for part, fields in controls.items():
            part_reference = reference.setdefault(part, {})
            for field, value in fields.items():
                part_reference.setdefault(field, float(value))

    calibrated: list[tuple[float, dict[str, dict[str, float]]]] = []
    for timestamp, controls in converted:
        relative_controls: dict[str, dict[str, float]] = {}
        for part, fields in controls.items():
            part_reference = reference.get(part, {})
            relative_controls[part] = {
                field: _shape_motion_delta(
                    part,
                    field,
                    _angle_difference(value, part_reference.get(field, value)),
                )
                for field, value in fields.items()
            }
        calibrated.append((timestamp, relative_controls))
    converted = calibrated
    source_times = [item[0] for item in converted]

    if duration_seconds is None:
        duration = source_times[-1]
    else:
        duration = max(float(duration_seconds), source_times[-1])
    if duration <= 0:
        duration = 1.0 / fps

    final_frame = max(1, int(round(duration * fps)))
    frame_times = [min(frame / fps, duration) for frame in range(final_frame + 1)]

    part_frame_values: dict[str, list[dict[str, float]]] = {}
    for part, fields in PART_FIELDS.items():
        part_sources: dict[str, tuple[list[float], list[float]]] = {}
        for field in fields:
            valid = [
                (timestamp, controls[part][field])
                for timestamp, controls in converted
                if part in controls and field in controls[part]
            ]
            if not valid:
                part_sources[field] = ([0.0], [0.0])
                continue
            values = _unwrap_degrees([item[1] for item in valid])
            part_sources[field] = ([item[0] for item in valid], values)

        frame_values = []
        for target in frame_times:
            values = {
                field: _interpolate(
                    part_sources[field][0],
                    part_sources[field][1],
                    target,
                )
                for field in fields
            }
            frame_values.append(values)

        for field in fields:
            filtered = _smooth(
                [frame[field] for frame in frame_values],
                smooth_motion,
                fps=int(fps),
            )
            for index, value in enumerate(filtered):
                frame_values[index][field] = value
        part_frame_values[part] = frame_values

    keyframe_counts: dict[str, int] = {}
    for timeline in data["timelines"]:
        if not isinstance(timeline, dict):
            continue
        timeline_type = timeline.get("type")
        if timeline_type == "folder":
            # Freeze the initial transform so the template's group offset is retained.
            _static_first_keyframe(timeline)
        elif timeline_type == "char":
            # Freeze the template's first character transform. This also removes
            # any root animation left inside a previously exported .miobject.
            _static_first_keyframe(timeline)
        elif timeline_type == "bodypart":
            part = timeline.get("model_part_name")
            if part in PART_FIELDS:
                fields = PART_FIELDS[part]
                values = part_frame_values[part]
                selected = _select_keyframes(values, fields, int(fps), optimize_keyframes)
                timeline["keyframes"] = {}
                for frame in selected:
                    keyframe_values = dict(static_part_fields.get(part, {}))
                    keyframe_values.update({
                        field: round(float(values[frame][field]), 4)
                        for field in fields
                    })
                    timeline["keyframes"][str(frame)] = keyframe_values
                keyframe_counts[part] = len(selected)
            else:
                # Keep accessories and other parts at their initial transform.
                _static_first_keyframe(timeline)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise MiobjectError(f"No se pudo guardar el archivo: {exc}") from exc

    return {
        "path": str(output),
        "duration": duration,
        "fps": int(fps),
        "final_frame": final_frame,
        "keyframe_counts": keyframe_counts,
        "total_keyframes": sum(keyframe_counts.values()),
    }
