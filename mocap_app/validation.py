from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ValidationProjectError(ValueError):
    """Raised when an animated asset cannot be assembled into a Mine-imator project."""


def build_validation_project(
    template_path: str | Path,
    animated_miobject_path: str | Path,
    output_path: str | Path,
    fps: int = 30,
    project_name: str = "Steve Motion Capture Validation",
) -> dict[str, Any]:
    """Combine an untouched Mine-imator scene with the animated humanoid asset.

    The original .miproject contributes editor, render and background settings.
    The exported .miobject contributes the matching rig, timelines and keyframes.
    This makes a directly openable validation scene without modifying the fixture.
    """
    if not 12 <= int(fps) <= 60:
        raise ValidationProjectError("Los FPS del proyecto deben estar entre 12 y 60.")

    base_path = Path(template_path)
    asset_path = Path(animated_miobject_path)
    destination = Path(output_path)
    try:
        base = json.loads(base_path.read_text(encoding="utf-8-sig"))
        asset = json.loads(asset_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationProjectError(f"No se pudo leer la plantilla o la animación: {exc}") from exc

    if not isinstance(base, dict) or not isinstance(base.get("project"), dict):
        raise ValidationProjectError("La plantilla debe ser un proyecto .miproject completo.")
    if not isinstance(asset, dict) or not isinstance(asset.get("timelines"), list):
        raise ValidationProjectError("La salida .miobject no contiene pistas de animación.")
    if not any(
        isinstance(item, dict)
        and isinstance(item.get("model"), dict)
        and item["model"].get("name") == "human"
        for item in asset.get("templates", [])
    ):
        raise ValidationProjectError("La animación exportada no contiene un modelo humano.")

    project = dict(base)
    for section in ("templates", "timelines", "lightgroups", "resources"):
        if section in asset:
            project[section] = asset[section]
    project_settings = dict(project["project"])
    project_settings["name"] = str(project_name)
    project_settings["tempo"] = int(fps)
    project["project"] = project_settings

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(project, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise ValidationProjectError(f"No se pudo guardar el proyecto de validación: {exc}") from exc

    return {
        "path": str(destination),
        "format": project.get("format"),
        "fps": int(fps),
        "timeline_count": len(project["timelines"]),
        "animated_parts": sum(
            1 for item in project["timelines"]
            if isinstance(item, dict)
            and item.get("type") == "bodypart"
            and bool(item.get("keyframes"))
        ),
    }
