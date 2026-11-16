from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path
from typing import Any, Iterable

from .retarget import calculate_pose_controls, stabilize_pose_samples


REQUIRED_PARTS = {"body", "head", "left_arm", "right_arm", "left_leg", "right_leg"}
PART_FIELDS = {
    "body": ("ROT_X", "ROT_Y", "ROT_Z", "BEND_ANGLE_X", "BEND_ANGLE_Z"),
    "head": ("ROT_X", "ROT_Y", "ROT_Z"),
    "left_arm": ("ROT_X", "ROT_Y", "ROT_Z", "BEND_ANGLE_X"),
    "right_arm": ("ROT_X", "ROT_Y", "ROT_Z", "BEND_ANGLE_X"),
    "left_leg": ("ROT_X", "ROT_Y", "ROT_Z", "BEND_ANGLE_X"),
    "right_leg": ("ROT_X", "ROT_Y", "ROT_Z", "BEND_ANGLE_X"),
    "root": ("ROT_X", "ROT_Y", "ROT_Z"),
}

KEYFRAME_ANGLE_THRESHOLD = 0.18
MAX_KEYFRAME_INTERVAL_SECONDS = 0.10


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


def _first_keyframe_values(timeline: dict[str, Any]) -> dict[str, float]:
    keyframes = timeline.get("keyframes")
    if not isinstance(keyframes, dict) or not keyframes:
        return {}
    try:
        first_frame = min(keyframes, key=lambda frame: float(frame))
    except (TypeError, ValueError):
        first_frame = next(iter(keyframes))
    values = keyframes.get(first_frame, {})
    if not isinstance(values, dict):
        return {}
    return {
        str(key): float(value)
        for key, value in values.items()
        if isinstance(value, (int, float))
    }


def _template_part_parents(data: dict[str, Any]) -> dict[str, str]:
    """Resolve the direct parent of every required body part."""
    timelines = [
        timeline
        for timeline in data.get("timelines", [])
        if isinstance(timeline, dict)
    ]
    by_id = {
        str(timeline.get("id")): timeline
        for timeline in timelines
        if timeline.get("id") is not None
    }
    defaults = {
        "body": "root",
        "head": "body",
        "left_arm": "body",
        "right_arm": "body",
        "left_leg": "root",
        "right_leg": "root",
    }
    parents: dict[str, str] = {}
    for timeline in timelines:
        if timeline.get("type") != "bodypart":
            continue
        part = timeline.get("model_part_name")
        if part not in REQUIRED_PARTS:
            continue
        parent_id = timeline.get("parent")
        parent = by_id.get(str(parent_id)) if parent_id is not None else None
        if parent is None:
            parents[part] = defaults[part]
        elif parent.get("type") == "char":
            parents[part] = "root"
        elif parent.get("type") == "bodypart":
            parent_part = parent.get("model_part_name")
            parents[part] = str(parent_part) if parent_part in REQUIRED_PARTS else defaults[part]
        else:
            parents[part] = defaults[part]
    return parents


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


def _template_body_bend_axes(data: dict[str, Any]) -> tuple[str, ...]:
    """Return the bend axes actually present in the selected torso rig."""
    body_timeline = next(
        (
            timeline
            for timeline in data.get("timelines", [])
            if isinstance(timeline, dict)
            and timeline.get("type") == "bodypart"
            and timeline.get("model_part_name") == "body"
        ),
        None,
    )
    if body_timeline is None:
        return ()
    values: dict[str, Any] = {}
    defaults = body_timeline.get("default_values")
    if isinstance(defaults, dict):
        values.update(defaults)
    values.update(_first_keyframe_data(body_timeline))
    return tuple(
        field
        for field in ("BEND_ANGLE_X", "BEND_ANGLE_Z")
        if field in values
    )


def _template_character_timeline(data: dict[str, Any]) -> dict[str, Any] | None:
    """Find the character timeline that directly owns the human body timeline."""
    timelines = [
        timeline
        for timeline in data.get("timelines", [])
        if isinstance(timeline, dict)
    ]
    by_id = {
        str(timeline.get("id")): timeline
        for timeline in timelines
        if timeline.get("id") is not None
    }
    body = next(
        (
            timeline for timeline in timelines
            if timeline.get("type") == "bodypart"
            and timeline.get("model_part_name") == "body"
        ),
        None,
    )
    if body is not None and body.get("parent") is not None:
        parent = by_id.get(str(body.get("parent")))
        if parent is not None and parent.get("type") == "char":
            return parent
    return next(
        (timeline for timeline in timelines if timeline.get("type") == "char"),
        None,
    )


def _template_root_baselines(data: dict[str, Any]) -> dict[str, float]:
    timeline = _template_character_timeline(data)
    if timeline is None:
        raise MiobjectError("La plantilla humana no contiene una pista de personaje (char).")
    values: dict[str, Any] = {}
    defaults = timeline.get("default_values")
    if isinstance(defaults, dict):
        values.update(defaults)
    values.update(_first_keyframe_data(timeline))
    return {
        field: float(values.get(field, 0.0))
        for field in PART_FIELDS["root"]
    }


def _template_root_static_fields(data: dict[str, Any]) -> dict[str, Any]:
    timeline = _template_character_timeline(data)
    if timeline is None:
        return {}
    # Keep only explicitly keyed properties here. default_values, such as the
    # character's scene position, remain in their original section untouched.
    values = _first_keyframe_data(timeline)
    return {
        str(key): value
        for key, value in values.items()
        if key not in PART_FIELDS["root"] and key != "TRANSITION"
    }


def _template_baselines(data: dict[str, Any]) -> dict[str, dict[str, float]]:
    baselines: dict[str, dict[str, float]] = {}
    for timeline in data.get("timelines", []):
        if not isinstance(timeline, dict) or timeline.get("type") != "bodypart":
            continue
        part = timeline.get("model_part_name")
        if part not in PART_FIELDS:
            continue

        defaults = timeline.get("default_values")
        baseline = {
            str(key): float(value)
            for key, value in (defaults.items() if isinstance(defaults, dict) else [])
            if isinstance(value, (int, float))
        }
        baseline.update(_first_keyframe_values(timeline))
        baselines[part] = {
            field: float(baseline.get(field, 0.0))
            for field in PART_FIELDS[part]
        }
    return baselines


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
    # Landmark filtering already removes most jitter. Keep this angle pass light
    # so slow, small movements are not swallowed by an overly strong low-pass.
    alpha = 0.78
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


def _reference_pose(samples: list[tuple[float, dict[str, Any]]]) -> dict[str, Any]:
    fallback = samples[0][1]
    for _, pose in samples:
        controls = calculate_pose_controls(pose, pose)
        if REQUIRED_PARTS.issubset(controls):
            return pose
    return fallback


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
    stabilized = stabilize_pose_samples(samples, enabled=smooth_motion)
    if not stabilized:
        raise MiobjectError("No hay muestras de pose para exportar.")

    reference_pose = _reference_pose(stabilized)
    baselines = _template_baselines(data)
    baselines["root"] = _template_root_baselines(data)
    character_timeline = _template_character_timeline(data)
    character_timeline_id = (
        str(character_timeline.get("id"))
        if character_timeline is not None and character_timeline.get("id") is not None
        else None
    )
    root_static_fields = _template_root_static_fields(data)
    part_parents = _template_part_parents(data)
    static_part_fields = _template_static_part_fields(data)
    body_bend_axes = _template_body_bend_axes(data)

    converted: list[tuple[float, dict[str, dict[str, float]]]] = []
    previous_controls: dict[str, dict[str, float]] = {}
    for timestamp, pose in stabilized:
        controls = calculate_pose_controls(
            pose,
            reference_pose=reference_pose,
            base_values=baselines,
            part_parents=part_parents,
            body_bend_axes=body_bend_axes,
            previous_controls=previous_controls,
        )
        if controls:
            converted.append((max(0.0, float(timestamp)), controls))
            previous_controls.update(controls)
    if not converted:
        raise MiobjectError(
            "No hay poses válidas para exportar. Comprueba que se vean los hombros y las caderas."
        )

    converted.sort(key=lambda item: item[0])
    deduplicated: dict[float, dict[str, dict[str, float]]] = {}
    for timestamp, controls in converted:
        deduplicated[timestamp] = controls
    converted = sorted(deduplicated.items(), key=lambda item: item[0])
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
                default_value = baselines.get(part, {}).get(field, 0.0)
                part_sources[field] = ([0.0], [default_value])
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
            if character_timeline_id is not None and str(timeline.get("id")) == character_timeline_id:
                fields = PART_FIELDS["root"]
                values = part_frame_values["root"]
                selected = _select_keyframes(values, fields, int(fps), optimize_keyframes)
                timeline["keyframes"] = {}
                for frame in selected:
                    keyframe_values = dict(root_static_fields)
                    keyframe_values.update({
                        field: round(float(values[frame][field]), 4)
                        for field in fields
                    })
                    timeline["keyframes"][str(frame)] = keyframe_values
                keyframe_counts["root"] = len(selected)
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
