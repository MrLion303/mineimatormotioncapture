import json
import math

from mocap_app.miobject import (
    _select_keyframes,
    _template_body_bend_axes,
    _template_part_parents,
    _template_static_part_fields,
    export_miobject,
)
from mocap_app.retarget import (
    _apply_rotation_delta,
    _quaternion_from_euler_xyz,
    _quaternion_multiply,
    _quaternion_to_euler_xyz,
    calculate_pose_controls,
)


def standing_pose():
    world = [[0.0, 0.0, 0.0, 1.0, 1.0] for _ in range(33)]
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
        world[index] = [*point, 1.0, 1.0]
    return {"world": world, "image": []}


def make_template(path):
    parts = {
        "body": {
            "ROT_X": 7.0,
            "ROT_Y": 4.4,
            "BEND_ANGLE_X": -4.4,
            "BEND_ANGLE_Z": 20.4,
        },
        "head": {"ROT_X": 3.2, "ROT_Y": 11.3},
        "left_arm": {"ROT_Y": 120.2, "BEND_ANGLE_X": 10.9, "POS_Z": -0.53},
        "right_arm": {"ROT_Y": -60.1, "BEND_ANGLE_X": 10.9, "POS_Z": -0.43},
        "left_leg": {"ROT_X": -29.5, "ROT_Y": 11.4, "BEND_ANGLE_X": 30.0},
        "right_leg": {"ROT_X": 37.1, "ROT_Y": -20.6, "BEND_ANGLE_X": 38.9},
    }
    timelines = [
        {"id": "folder", "type": "folder", "keyframes": {"0": {"ROT_Y": 80}, "30": {"ROT_Y": 120}}},
        {
            "id": "character",
            "type": "char",
            "default_values": {"POS_X": 12.0, "POS_Y": 24.0, "POS_Z": 3.0},
            "keyframes": {"0": {"ROT_Z": -80.0}},
        },
    ]
    part_ids = {part: f"part-{index}" for index, part in enumerate(parts)}
    for index, (part, values) in enumerate(parts.items()):
        parent = (
            part_ids["body"]
            if part in {"head", "left_arm", "right_arm"}
            else "character"
        )
        timelines.append({
            "id": part_ids[part],
            "type": "bodypart",
            "model_part_name": part,
            "parent": parent,
            "keyframes": {
                "0": {
                    **values,
                    **({"TEXTURE_OBJ": "GREGOR_HEAD_TEXTURE"} if part == "head" else {}),
                }
            },
        })
    timelines.append({
        "id": "hat",
        "type": "bodypart",
        "model_part_name": "hat",
        "keyframes": {},
    })
    template = {
        "format": 35,
        "created_in": "2.0.2 Community Build 1.0.0",
        "templates": [{"model": {"name": "human", "state": {"variant": "steve", "type": "slim"}}}],
        "timelines": timelines,
        "resources": [],
    }
    path.write_text(json.dumps(template), encoding="utf-8")


def test_retarget_produces_six_human_parts_and_rest_pose():
    pose = standing_pose()
    baselines = {
        "body": {"ROT_X": 7.0, "ROT_Y": 4.4, "ROT_Z": 0.0},
        "head": {"ROT_X": 3.2, "ROT_Y": 11.3, "ROT_Z": 0.0},
        "left_arm": {"ROT_X": 0.0, "ROT_Y": 120.2, "ROT_Z": 0.0, "BEND_ANGLE_X": 10.9},
        "right_arm": {"ROT_X": 0.0, "ROT_Y": -60.1, "ROT_Z": 0.0, "BEND_ANGLE_X": 10.9},
        "left_leg": {"ROT_X": -29.5, "ROT_Y": 11.4, "ROT_Z": 0.0, "BEND_ANGLE_X": 30.0},
        "right_leg": {"ROT_X": 37.1, "ROT_Y": -20.6, "ROT_Z": 0.0, "BEND_ANGLE_X": 38.9},
    }
    controls = calculate_pose_controls(pose, pose, baselines)
    assert {"body", "head", "left_arm", "right_arm", "left_leg", "right_leg"} <= set(controls)
    assert abs(controls["left_arm"]["ROT_Y"] - 120.2) < 0.01
    assert abs(controls["right_arm"]["ROT_Y"] + 60.1) < 0.01
    assert abs(controls["left_arm"]["BEND_ANGLE_X"] - 10.9) < 0.01


def test_arm_motion_is_relative_to_body_and_keeps_rest_offset():
    base = standing_pose()
    moved = standing_pose()
    moved["world"][13][0] += 0.12
    moved["world"][15][0] += 0.12

    baseline = {"left_arm": {"ROT_Y": 120.0, "BEND_ANGLE_X": 10.0}}
    controls = calculate_pose_controls(moved, base, baseline)
    assert "left_arm" in controls
    assert max(
        abs(controls["left_arm"][field] - baseline["left_arm"].get(field, 0.0))
        for field in ("ROT_X", "ROT_Y", "ROT_Z")
    ) > 0.1


def test_slow_subtle_motion_keeps_keyframes():
    frames = [{"ROT_X": index * 0.05} for index in range(16)]
    selected = _select_keyframes(frames, ("ROT_X",), fps=30, optimize=True)
    assert len(selected) >= 5
    assert selected[0] == 0
    assert selected[-1] == 15
    assert max(b - a for a, b in zip(selected, selected[1:])) <= 3


def test_export_preserves_template_rest_pose_and_group_transform(tmp_path):
    template_path = tmp_path / "template.miobject"
    output_path = tmp_path / "captured.miobject"
    make_template(template_path)

    pose_a = standing_pose()
    pose_b = standing_pose()
    pose_b["world"][13][0] += 0.12
    pose_b["world"][15][0] += 0.12
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
    assert exported["format"] == 35
    assert exported["templates"][0]["model"]["name"] == "human"
    character = next(item for item in exported["timelines"] if item["type"] == "char")
    assert abs(character["keyframes"]["0"]["ROT_Z"] - (-80.0)) < 0.02
    assert abs(character["keyframes"]["30"]["ROT_Z"] - (-80.0)) < 0.02
    assert character["default_values"] == {
        "POS_X": 12.0, "POS_Y": 24.0, "POS_Z": 3.0
    }
    assert character["keyframes"]["0"]["POS_X"] == 12.0
    assert character["keyframes"]["0"]["POS_Y"] == 24.0
    assert character["keyframes"]["0"]["POS_Z"] == 3.0
    folder = next(item for item in exported["timelines"] if item["type"] == "folder")
    assert folder["keyframes"] == {"0": {"ROT_Y": 80.0}}

    body = next(
        item for item in exported["timelines"]
        if item.get("model_part_name") == "body"
    )
    body_start = body["keyframes"]["0"]
    assert abs(body_start["BEND_ANGLE_X"] - (-4.4)) < 0.02
    assert abs(body_start["BEND_ANGLE_Z"] - 20.4) < 0.02

    head = next(
        item for item in exported["timelines"]
        if item.get("model_part_name") == "head"
    )
    assert head["keyframes"]["0"]["TEXTURE_OBJ"] == "GREGOR_HEAD_TEXTURE"

    left_arm = next(
        item for item in exported["timelines"]
        if item.get("model_part_name") == "left_arm"
    )
    right_arm = next(
        item for item in exported["timelines"]
        if item.get("model_part_name") == "right_arm"
    )
    assert left_arm["keyframes"]["0"]["POS_Z"] == -0.53
    assert right_arm["keyframes"]["0"]["POS_Z"] == -0.43

    expected_baselines = {
        "left_arm": {"ROT_Y": 120.2, "BEND_ANGLE_X": 10.9},
        "right_arm": {"ROT_Y": -60.1, "BEND_ANGLE_X": 10.9},
    }
    for part, expected in expected_baselines.items():
        timeline = next(
            item for item in exported["timelines"]
            if item.get("model_part_name") == part
        )
        start = timeline["keyframes"]["0"]
        for field, value in expected.items():
            assert abs(start[field] - value) < 0.02

    for part in ("body", "head", "left_arm", "right_arm", "left_leg", "right_leg"):
        timeline = next(
            item for item in exported["timelines"]
            if item.get("model_part_name") == part
        )
        assert "0" in timeline["keyframes"]
        assert "30" in timeline["keyframes"]
    assert result["total_keyframes"] == 7 * 31
    assert result["keyframe_counts"]["root"] == 31


def test_export_rejects_empty_capture(tmp_path):
    import pytest
    from mocap_app.miobject import MiobjectError

    template_path = tmp_path / "template.miobject"
    make_template(template_path)
    with pytest.raises(MiobjectError):
        export_miobject(template_path, tmp_path / "out.miobject", [], fps=30)


def rotate_pose_y(pose, angle_degrees):
    angle = math.radians(angle_degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotated = {key: value for key, value in pose.items()}
    rotated_world = []
    for point in pose["world"]:
        x, y, z, *rest = point
        rotated_world.append([
            x * cosine + z * sine,
            y,
            -x * sine + z * cosine,
            *rest,
        ])
    rotated["world"] = rotated_world
    return rotated


def test_template_hierarchy_detects_body_children_and_root_legs(tmp_path):
    template_path = tmp_path / "template.miobject"
    make_template(template_path)
    from mocap_app.miobject import load_template

    template = load_template(template_path)
    parents = _template_part_parents(template)
    assert parents["body"] == "root"
    assert parents["head"] == "body"
    assert parents["left_arm"] == "body"
    assert parents["right_arm"] == "body"
    assert parents["left_leg"] == "root"
    assert parents["right_leg"] == "root"
    assert _template_body_bend_axes(template) == ("BEND_ANGLE_X", "BEND_ANGLE_Z")
    static_fields = _template_static_part_fields(template)
    assert static_fields["head"]["TEXTURE_OBJ"] == "GREGOR_HEAD_TEXTURE"
    assert static_fields["left_arm"]["POS_Z"] == -0.53
    assert static_fields["right_arm"]["POS_Z"] == -0.43


def test_torso_only_twist_stays_on_body_and_does_not_turn_character_root():
    neutral = standing_pose()
    turned = standing_pose()
    angle = math.radians(65.0)
    cosine = math.cos(angle)
    sine = math.sin(angle)

    # Rotate the upper body landmarks only; pelvis and legs remain fixed.
    for index in (0, 7, 8, 11, 12, 13, 14, 15, 16):
        x, y, z, *rest = neutral["world"][index]
        turned["world"][index] = [
            x * cosine + z * sine,
            y,
            -x * sine + z * cosine,
            *rest,
        ]

    parents = {
        "body": "root",
        "head": "body",
        "left_arm": "body",
        "right_arm": "body",
        "left_leg": "root",
        "right_leg": "root",
    }
    controls = calculate_pose_controls(turned, neutral, part_parents=parents)
    assert abs(controls["root"]["ROT_Z"]) < 0.5
    assert abs(controls["body"]["ROT_Y"]) > 20.0


def test_full_body_turn_rotates_body_and_root_parented_legs_together():
    neutral = standing_pose()
    turned = rotate_pose_y(neutral, 55.0)
    parents = {
        "body": "root",
        "head": "body",
        "left_arm": "body",
        "right_arm": "body",
        "left_leg": "root",
        "right_leg": "root",
    }
    controls = calculate_pose_controls(turned, neutral, part_parents=parents)
    assert abs(abs(controls["root"]["ROT_Z"]) - 55.0) < 0.5
    assert abs(controls["body"]["ROT_Y"]) < 0.5
    assert abs(controls["left_leg"]["ROT_Y"]) < 0.5
    assert abs(controls["right_leg"]["ROT_Y"]) < 0.5



def test_leg_swing_does_not_rotate_the_character_root():
    neutral = standing_pose()
    walking = standing_pose()
    walking["world"][25][0] -= 0.18
    walking["world"][27][0] -= 0.30
    walking["world"][26][0] += 0.16
    walking["world"][28][0] += 0.28
    controls = calculate_pose_controls(walking, neutral)
    assert abs(controls["root"]["ROT_Z"]) < 0.01
    for part in ("left_leg", "right_leg"):
        assert max(
            abs(controls[part][field])
            for field in ("ROT_X", "ROT_Y", "ROT_Z", "BEND_ANGLE_X")
        ) > 1.0


def _with_image_projection(pose, shift_x=0.0, shift_y=0.0):
    projected = {key: value for key, value in pose.items()}
    image = []
    for point in pose["world"]:
        image.append([
            0.5 + point[0] * 0.1 + shift_x,
            0.5 + point[1] * 0.1 + shift_y,
            point[2] * 0.1,
            1.0,
            1.0,
        ])
    projected["image"] = image
    projected["image_size"] = (640, 480)
    return projected


def test_root_translation_tracks_screen_displacement_and_can_be_disabled():
    neutral = _with_image_projection(standing_pose())
    moved = _with_image_projection(standing_pose(), shift_x=0.05, shift_y=-0.04)
    baseline = {"root": {"POS_X": 12.0, "POS_Y": 24.0, "POS_Z": 3.0}}

    controls = calculate_pose_controls(
        moved,
        neutral,
        base_values=baseline,
        translation_scale=18.0,
    )
    assert controls["root"]["POS_X"] > 12.0
    assert controls["root"]["POS_Z"] > 3.0
    assert abs(controls["root"]["POS_Y"] - 24.0) < 0.1

    frozen = calculate_pose_controls(
        moved,
        neutral,
        base_values=baseline,
        capture_translation=False,
    )
    assert frozen["root"]["POS_X"] == 12.0
    assert frozen["root"]["POS_Y"] == 24.0
    assert frozen["root"]["POS_Z"] == 3.0


def test_limb_orientation_uses_the_bend_plane_without_losing_rest_rotation():
    neutral = standing_pose()
    moved = standing_pose()
    moved["world"][13][2] += 0.20
    moved["world"][15][2] += 0.36
    baseline = {"left_arm": {"ROT_X": 4.0, "ROT_Y": 120.0, "ROT_Z": -8.0}}
    controls = calculate_pose_controls(moved, neutral, base_values=baseline)
    assert max(
        abs(controls["left_arm"][field] - baseline["left_arm"].get(field, 0.0))
        for field in ("ROT_X", "ROT_Y", "ROT_Z")
    ) > 0.1


def test_euler_conversion_stays_continuous_when_yaw_crosses_90_degrees():
    pose_80 = _quaternion_from_euler_xyz((0.0, 80.0, 0.0))
    pose_100 = _quaternion_from_euler_xyz((0.0, 100.0, 0.0))
    previous = _quaternion_to_euler_xyz(pose_80)
    current = _quaternion_to_euler_xyz(pose_100, reference_angles=previous)

    assert abs(current[0]) < 0.1
    assert abs(current[1] - 100.0) < 0.1
    assert abs(current[2]) < 0.1


def test_export_places_global_facing_rotation_on_char_root(tmp_path):
    template_path = tmp_path / "template.miobject"
    output_path = tmp_path / "turned.miobject"
    make_template(template_path)

    neutral = standing_pose()
    turned = rotate_pose_y(neutral, 55.0)
    export_miobject(
        template_path,
        output_path,
        [(0.0, neutral), (1.0, turned)],
        fps=30,
        duration_seconds=1.0,
        optimize_keyframes=False,
        smooth_motion=False,
    )

    exported = json.loads(output_path.read_text(encoding="utf-8"))
    character = next(item for item in exported["timelines"] if item["type"] == "char")
    body = next(
        item for item in exported["timelines"]
        if item.get("model_part_name") == "body"
    )

    first_root_yaw = character["keyframes"]["0"]["ROT_Z"]
    last_root_yaw = character["keyframes"]["30"]["ROT_Z"]
    first_body_yaw = body["keyframes"]["0"]["ROT_Y"]
    last_body_yaw = body["keyframes"]["30"]["ROT_Y"]

    assert abs(abs(last_root_yaw - first_root_yaw) - 55.0) < 0.5
    assert abs(last_body_yaw - first_body_yaw) < 0.5



def test_local_joint_delta_is_composed_with_rest_rotation_in_local_order():
    baseline = {
        "left_arm": {
            "ROT_X": 27.0,
            "ROT_Y": 113.0,
            "ROT_Z": -18.0,
        }
    }
    delta = _quaternion_from_euler_xyz((13.0, -7.0, 11.0))
    result = _apply_rotation_delta(baseline, "left_arm", delta)
    actual = _quaternion_from_euler_xyz((
        result["ROT_X"], result["ROT_Y"], result["ROT_Z"]
    ))
    rest = _quaternion_from_euler_xyz((27.0, 113.0, -18.0))
    expected = _quaternion_multiply(rest, delta)
    # q and -q represent the same orientation.
    similarity = abs(sum(a * b for a, b in zip(actual, expected)))
    assert abs(similarity - 1.0) < 1e-5



def test_export_writes_camera_translation_to_char_position_tracks(tmp_path):
    template_path = tmp_path / "template.miobject"
    output_path = tmp_path / "translated.miobject"
    make_template(template_path)

    neutral = _with_image_projection(standing_pose())
    moved = _with_image_projection(standing_pose(), shift_x=0.04, shift_y=-0.03)
    export_miobject(
        template_path,
        output_path,
        [(0.0, neutral), (1.0, moved)],
        fps=30,
        duration_seconds=1.0,
        optimize_keyframes=False,
        smooth_motion=False,
        translation_scale=18.0,
    )

    exported = json.loads(output_path.read_text(encoding="utf-8"))
    character = next(item for item in exported["timelines"] if item["type"] == "char")
    first = character["keyframes"]["0"]
    last = character["keyframes"]["30"]
    assert last["POS_X"] > first["POS_X"]
    assert last["POS_Z"] > first["POS_Z"]
