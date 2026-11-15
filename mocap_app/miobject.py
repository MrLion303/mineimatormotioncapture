from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path
from typing import Any, Iterable

from .retarget import calculate_pose_controls


REQUIRED_PARTS = {"body", "head", "left_arm", "right_arm", "left_leg", "right_leg"}
PART_FIELDS = {
    "body": ("ROT_X", "ROT_Y", "ROT_Z"),
    "head": ("ROT_X", "ROT_Y", "ROT_Z"),
    "left_arm": ("ROT_X", "ROT_Y", "ROT_Z", "BEND_ANGLE_X"),
    "right_arm": ("ROT_X", "ROT_Y", "ROT_Z", "BEND_ANGLE_X"),
    "left_leg": ("ROT_X", "ROT_Y", "ROT_Z", "BEND_ANGLE_X"),
    "right_leg": ("ROT_X", "ROT_Y", "ROT_Z", "BEND_ANGLE_X"),
}


class MiobjectError(ValueError):
    pass


def load_template(path: str | Path) -> dict[str, Any]:
    template_path = Path(path)
    try:
        with template_path.open("r", encoding="utf-8-sig") as source:
            data = json.load(source)
    except (OSError, json.JSONDecodeError) as exc:
        raise MiobjectError(f"No se pudo leer el archivo .miobject: {exc}") from exc
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


def _relative_degrees(value: float, reference: float) -> float:
    delta = value - reference
    while delta > 180.0:
        delta -= 360.0
    while delta < -180.0:
        delta += 360.0
    return delta


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


def _smooth(values: list[float], enabled: bool) -> list[float]:
    if not enabled or len(values) < 3:
        return values
    alpha = 0.38
    output = [values[0]]
    for value in values[1:]:
        output.append(output[-1] + alpha * (value - output[-1]))
    return output


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

    threshold = 1.0
    max_gap = max(2, round(fps * 0.20))
    selected = [0]
    previous = 0
    for index in range(1, len(frame_values) - 1):
        difference = max(
            abs(frame_values[index][field] - frame_values[previous][field])
            for field in fields
        )
        if difference >= threshold or index - previous >= max_gap:
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
    smooth_motion: bool = True,
) -> dict[str, Any]:
    if not 12 <= int(fps) <= 60:
        raise MiobjectError("Los FPS de salida deben estar entre 12 y 60.")

    data = load_template(template_path)
    converted: list[tuple[float, dict[str, dict[str, float]]]] = []
    for timestamp, pose in samples:
        controls = calculate_pose_controls(pose)
        if controls:
            converted.append((max(0.0, float(timestamp)), controls))
    if not converted:
        raise MiobjectError("No hay poses válidas para exportar. Revisa que se vea el cuerpo completo.")

    converted.sort(key=lambda item: item[0])
    deduplicated: dict[float, dict[str, dict[str, float]]] = {}
    for timestamp, controls in converted:
        deduplicated[timestamp] = controls
    converted = sorted(deduplicated.items(), key=lambda item: item[0])
    source_times = [item[0] for item in converted]

    # Treat the first valid captured pose as the neutral reference pose.
    references: dict[str, dict[str, float]] = {}
    for _, controls in converted:
        for part, values in controls.items():
            if part not in references:
                references[part] = dict(values)
    normalized = []
    for timestamp, controls in converted:
        relative = {}
        for part, values in controls.items():
            reference = references.get(part, {})
            relative[part] = {
                field: _relative_degrees(value, reference.get(field, value))
                for field, value in values.items()
            }
        normalized.append((timestamp, relative))
    converted = normalized
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
                field: _interpolate(part_sources[field][0], part_sources[field][1], target)
                for field in fields
            }
            frame_values.append(values)

        for field in fields:
            filtered = _smooth([frame[field] for frame in frame_values], smooth_motion)
            for index, value in enumerate(filtered):
                frame_values[index][field] = value
        part_frame_values[part] = frame_values

    keyframe_counts: dict[str, int] = {}
    for timeline in data["timelines"]:
        if not isinstance(timeline, dict):
            continue
        timeline_type = timeline.get("type")
        if timeline_type == "folder":
            # Remove animation inherited from a sample pose; keep the hierarchy and group settings.
            timeline["keyframes"] = {}
        elif timeline_type == "bodypart":
            part = timeline.get("model_part_name")
            if part in PART_FIELDS:
                fields = PART_FIELDS[part]
                values = part_frame_values[part]
                selected = _select_keyframes(values, fields, int(fps), optimize_keyframes)
                timeline["keyframes"] = {
                    str(frame): {
                        field: round(float(values[frame][field]), 4)
                        for field in fields
                    }
                    for frame in selected
                }
                keyframe_counts[part] = len(selected)
            elif part == "hat":
                # The hat inherits the head transform and needs no duplicate animation.
                timeline["keyframes"] = {}

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
