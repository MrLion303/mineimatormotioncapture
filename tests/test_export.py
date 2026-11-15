import json

from mocap_app.miobject import export_miobject
from mocap_app.retarget import calculate_pose_controls


def standing_pose():
    world = [[0.0, 0.0, 0.0, 1.0] for _ in range(33)]
    values = {
        0: (0.0, -1.9, -0.1),
        7: (-0.12, -1.75, 0.0),
        8: (0.12, -1.75, 0.0),
        11: (-0.25, -1.25, 0.0),
        12: (0.25, -1.25, 0.0),
        13: (-0.28, -0.85, 0.0),
        14: (0.28, -0.85, 0.0),
        15: (-0.30, -0.45, 0.0),
        16: (0.30, -0.45, 0.0),
        23: (-0.16, 0.0, 0.0),
        24: (0.16, 0.0, 0.0),
        25: (-0.17, 0.75, 0.0),
        26: (0.17, 0.75, 0.0),
        27: (-0.17, 1.5, 0.0),
        28: (0.17, 1.5, 0.0),
    }
    for index, point in values.items():
        world[index] = [*point, 1.0]
    return {"world": world, "image": []}


def make_template(path):
    parts = ["body", "head", "left_arm", "right_arm", "left_leg", "right_leg", "hat"]
    timelines = [
        {"id": "folder", "type": "folder", "keyframes": {"0": {"ROT_Y": 80}}},
        {"id": "character", "type": "char", "keyframes": {"0": {"POS_X": 0}}},
    ]
    for index, part in enumerate(parts):
        timelines.append({
            "id": f"part-{index}",
            "type": "bodypart",
            "model_part_name": part,
            "keyframes": {"0": {"ROT_X": 99}},
        })
    template = {
        "format": 34,
        "created_in": "2.0 (Pre-release 7)",
        "templates": [{"model": {"name": "human", "state": {"variant": "steve"}}}],
        "timelines": timelines,
        "resources": [],
    }
    path.write_text(json.dumps(template), encoding="utf-8")


def test_retarget_produces_six_human_parts():
    controls = calculate_pose_controls(standing_pose())
    assert {"body", "head", "left_arm", "right_arm", "left_leg", "right_leg"} <= set(controls)
    assert "BEND_ANGLE_X" in controls["left_arm"]
    assert "BEND_ANGLE_X" in controls["right_leg"]


def test_export_preserves_template_and_writes_keyframes(tmp_path):
    template_path = tmp_path / "template.miobject"
    output_path = tmp_path / "captured.miobject"
    make_template(template_path)

    pose_a = standing_pose()
    pose_b = standing_pose()
    pose_b["world"][15][0] = -0.65
    samples = [(0.0, pose_a), (0.5, pose_b), (1.0, pose_a)]

    result = export_miobject(
        template_path,
        output_path,
        samples,
        fps=30,
        duration_seconds=1.0,
        optimize_keyframes=False,
        smooth_motion=False,
    )

    exported = json.loads(output_path.read_text(encoding="utf-8"))
    assert exported["format"] == 34
    assert exported["templates"][0]["model"]["name"] == "human"
    folder = next(item for item in exported["timelines"] if item["type"] == "folder")
    assert folder["keyframes"] == {}
    for part in ("body", "head", "left_arm", "right_arm", "left_leg", "right_leg"):
        timeline = next(
            item for item in exported["timelines"]
            if item.get("model_part_name") == part
        )
        assert "0" in timeline["keyframes"]
        assert "30" in timeline["keyframes"]
    assert result["total_keyframes"] == 6 * 31


def test_export_rejects_empty_capture(tmp_path):
    import pytest
    from mocap_app.miobject import MiobjectError

    template_path = tmp_path / "template.miobject"
    make_template(template_path)
    with pytest.raises(MiobjectError):
        export_miobject(template_path, tmp_path / "out.miobject", [], fps=30)
