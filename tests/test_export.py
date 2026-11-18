import json

import pytest

from mocap_app.miobject import (
    MiobjectError,
    _select_keyframes,
    _shape_motion_delta,
    _smooth,
    export_miobject,
)
from mocap_app.retarget import (
    WORLD_REFERENCE_FRAME,
    _segment_rotation,
    _torso_frame,
    calculate_pose_controls,
)


def standing_pose():
    world = [[0.0, 0.0, 0.0, 1.0, 1.0] for _ in range(33)]
    points = {
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
    for index, point in points.items():
        world[index] = [*point, 1.0, 1.0]
    return {"world": world, "image": []}


def make_template(path):
    parts = {
        "body": {
            "ROT_X": 0.0,
            "ROT_Y": 0.0,
            "ROT_Z": 0.0,
            "BEND_ANGLE_X": -4.4,
            "BEND_ANGLE_Y": 2.0,
            "BEND_ANGLE_Z": 20.4,
        },
        "head": {"ROT_X": 0.0, "ROT_Y": 0.0, "ROT_Z": 0.0},
        "left_arm": {
            "ROT_X": 0.0, "ROT_Y": 0.0, "ROT_Z": 0.0,
            "BEND_ANGLE_X": 0.0, "POS_Z": -0.53,
        },
        "right_arm": {
            "ROT_X": 0.0, "ROT_Y": 0.0, "ROT_Z": 0.0,
            "BEND_ANGLE_X": 0.0, "POS_Z": -0.43,
        },
        "left_leg": {
            "ROT_X": 0.0, "ROT_Y": 0.0, "ROT_Z": 0.0,
            "BEND_ANGLE_X": 0.0,
        },
        "right_leg": {
            "ROT_X": 0.0, "ROT_Y": 0.0, "ROT_Z": 0.0,
            "BEND_ANGLE_X": 0.0,
        },
    }
    timelines = [
        {
            "id": "folder",
            "type": "folder",
            "keyframes": {"0": {"ROT_Y": 80}, "30": {"ROT_Y": 120}},
        },
        {
            "id": "character",
            "type": "char",
            "default_values": {"POS_X": 12.0, "POS_Y": 24.0, "POS_Z": 3.0},
            "keyframes": {
                "0": {"POS_X": 12.0, "POS_Y": 24.0, "POS_Z": 3.0, "ROT_Z": 0.0},
                "30": {"POS_X": 80.0, "POS_Y": 90.0, "POS_Z": 45.0, "ROT_Z": 90.0},
            },
        },
    ]
    part_ids = {part: f"part-{index}" for index, part in enumerate(parts)}
    for part, values in parts.items():
        parent = (
            part_ids["body"]
            if part in {"head", "left_arm", "right_arm"}
            else "character"
        )
        first = dict(values)
        if part == "head":
            first["TEXTURE_OBJ"] = "GREGOR_HEAD_TEXTURE"
        timelines.append({
            "id": part_ids[part],
            "type": "bodypart",
            "model_part_name": part,
            "parent": parent,
            "keyframes": {"0": first},
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
        "templates": [{
            "model": {
                "name": "human",
                "state": {"variant": "steve", "type": "wide"},
            }
        }],
        "timelines": timelines,
        "resources": [],
    }
    path.write_text(json.dumps(template), encoding="utf-8")


def timeline_for(data, part):
    return next(
        item for item in data["timelines"]
        if item.get("model_part_name") == part
    )


def test_direct_retarget_uses_the_native_channels_from_the_mineimator_example():
    controls = calculate_pose_controls(standing_pose())

    assert set(controls) == {
        "body", "head", "left_arm", "right_arm", "left_leg", "right_leg"
    }
    assert set(controls["body"]) == {"ROT_X", "ROT_Y", "ROT_Z"}
    assert set(controls["head"]) == {"ROT_X", "ROT_Z"}

    for part in ("left_arm", "right_arm", "left_leg", "right_leg"):
        assert set(controls[part]) == {
            "ROT_X", "ROT_Y", "ROT_Z", "BEND_ANGLE_X"
        }
        assert abs(controls[part]["ROT_Y"]) < 0.001

    for part in ("left_leg", "right_leg"):
        assert abs(controls[part]["ROT_X"]) < 0.01
        assert abs(controls[part]["ROT_Z"]) < 1.0
        assert abs(controls[part]["BEND_ANGLE_X"]) < 1.0


def test_left_and_right_legs_follow_their_own_captured_segments():
    neutral = standing_pose()
    moved = standing_pose()
    moved["world"][25][0] -= 0.16
    moved["world"][27][0] -= 0.30
    moved["world"][26][0] += 0.14
    moved["world"][28][0] += 0.27
    moved["world"][27][1] += 0.12
    moved["world"][28][1] -= 0.08

    original = calculate_pose_controls(neutral)
    controls = calculate_pose_controls(moved)
    for part in ("left_leg", "right_leg"):
        assert abs(controls[part]["ROT_Z"] - original[part]["ROT_Z"]) > 5.0
        assert 0.0 <= controls[part]["BEND_ANGLE_X"] <= 150.0
    assert "root" not in controls
    assert "char" not in controls


def test_arm_segments_and_elbow_bends_are_mapped_independently():
    neutral = standing_pose()
    moved = standing_pose()
    moved["world"][13][2] += 0.22
    moved["world"][15][2] += 0.40
    moved["world"][14][2] -= 0.18
    moved["world"][16][2] -= 0.28

    original = calculate_pose_controls(neutral)
    controls = calculate_pose_controls(moved)
    assert abs(controls["left_arm"]["ROT_X"] - original["left_arm"]["ROT_X"]) > 1.0
    assert abs(controls["right_arm"]["ROT_X"] - original["right_arm"]["ROT_X"]) > 1.0
    assert controls["left_arm"]["BEND_ANGLE_X"] > original["left_arm"]["BEND_ANGLE_X"]
    assert controls["right_arm"]["BEND_ANGLE_X"] > original["right_arm"]["BEND_ANGLE_X"]


def test_torso_and_head_use_their_own_native_landmark_axes():
    neutral = standing_pose()
    moved = standing_pose()
    moved["world"][11][2] += 0.18
    moved["world"][12][2] += 0.18
    moved["world"][0][1] -= 0.10
    moved["world"][7][1] += 0.04
    moved["world"][8][1] -= 0.04

    original = calculate_pose_controls(neutral)
    controls = calculate_pose_controls(moved)
    assert any(
        abs(controls["body"][field] - original["body"][field]) > 1.0
        for field in ("BEND_ANGLE_X", "BEND_ANGLE_Y", "BEND_ANGLE_Z")
    )
    assert (
        abs(controls["head"]["ROT_X"] - original["head"]["ROT_X"]) > 1.0
        or abs(controls["head"]["ROT_Z"] - original["head"]["ROT_Z"]) > 1.0
    )


def test_slow_subtle_motion_keeps_keyframes():
    frames = [{"ROT_X": index * 0.05} for index in range(16)]
    selected = _select_keyframes(frames, ("ROT_X",), fps=30, optimize=True)
    assert len(selected) >= 4
    assert selected[0] == 0
    assert selected[-1] == 15
    assert max(b - a for a, b in zip(selected, selected[1:])) <= 6


def test_export_calibrates_starting_stance_and_uses_native_channels(tmp_path):
    template_path = tmp_path / "template.miobject"
    output_path = tmp_path / "captured.miobject"
    make_template(template_path)

    pose_a = standing_pose()
    pose_b = standing_pose()
    pose_b["world"][11][2] += 0.18
    pose_b["world"][12][2] += 0.18
    pose_b["world"][13][2] += 0.22
    pose_b["world"][15][2] += 0.40
    pose_b["world"][25][0] -= 0.16
    pose_b["world"][27][0] -= 0.30
    pose_b["world"][26][0] += 0.14
    pose_b["world"][28][0] += 0.27
    pose_b["world"][0][1] -= 0.10
    pose_b["world"][7][1] += 0.04
    pose_b["world"][8][1] -= 0.04

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
    character = next(item for item in exported["timelines"] if item["type"] == "char")
    assert character["default_values"] == {
        "POS_X": 12.0, "POS_Y": 24.0, "POS_Z": 3.0
    }
    assert character["keyframes"] == {
        "0": {"POS_X": 12.0, "POS_Y": 24.0, "POS_Z": 3.0, "ROT_Z": 0.0}
    }

    for part in ("body", "head", "left_arm", "right_arm", "left_leg", "right_leg"):
        timeline = timeline_for(exported, part)
        assert len(timeline["keyframes"]) == 31
        assert "POS_X" not in timeline["keyframes"]["0"]
        assert "POS_Y" not in timeline["keyframes"]["0"]

    body = timeline_for(exported, "body")
    first_body = body["keyframes"]["0"]
    assert all(abs(first_body[field]) < 0.001 for field in (
        "ROT_X", "ROT_Y", "ROT_Z"
    ))
    # Torso orientation uses rigid rotation; template bend deformation stays fixed.
    assert first_body["BEND_ANGLE_X"] == -4.4
    assert first_body["BEND_ANGLE_Y"] == 2.0
    assert first_body["BEND_ANGLE_Z"] == 20.4
    for field in ("BEND_ANGLE_X", "BEND_ANGLE_Y", "BEND_ANGLE_Z"):
        assert len({frame[field] for frame in body["keyframes"].values()}) == 1

    head = timeline_for(exported, "head")
    first_head = head["keyframes"]["0"]
    assert abs(first_head["ROT_X"]) < 0.001
    assert abs(first_head["ROT_Z"]) < 0.001
    assert len({frame["ROT_Y"] for frame in head["keyframes"].values()}) == 1

    for part in ("left_arm", "right_arm", "left_leg", "right_leg"):
        first = timeline_for(exported, part)["keyframes"]["0"]
        assert abs(first["ROT_X"]) < 0.001
        assert abs(first["ROT_Z"]) < 0.001
        assert abs(first["BEND_ANGLE_X"]) < 0.001

    left_leg = timeline_for(exported, "left_leg")
    right_leg = timeline_for(exported, "right_leg")
    assert any(abs(frame["ROT_X"]) + abs(frame["ROT_Z"]) > 1.0
               for frame in left_leg["keyframes"].values())
    assert any(abs(frame["ROT_X"]) + abs(frame["ROT_Z"]) > 1.0
               for frame in right_leg["keyframes"].values())

    assert timeline_for(exported, "left_arm")["keyframes"]["0"]["POS_Z"] == -0.53
    assert timeline_for(exported, "right_arm")["keyframes"]["0"]["POS_Z"] == -0.43
    assert result["keyframe_counts"] == {
        "body": 31,
        "head": 31,
        "left_arm": 31,
        "right_arm": 31,
        "left_leg": 31,
        "right_leg": 31,
    }
    assert result["total_keyframes"] == 6 * 31


def test_export_preserves_static_rig_properties_and_accessories(tmp_path):
    template_path = tmp_path / "template.miobject"
    output_path = tmp_path / "captured.miobject"
    make_template(template_path)

    export_miobject(
        template_path,
        output_path,
        [(0.0, standing_pose()), (1.0, standing_pose())],
        fps=30,
        duration_seconds=1.0,
        optimize_keyframes=False,
        smooth_motion=False,
    )

    exported = json.loads(output_path.read_text(encoding="utf-8"))
    assert timeline_for(exported, "head")["keyframes"]["0"]["TEXTURE_OBJ"] == "GREGOR_HEAD_TEXTURE"
    assert timeline_for(exported, "left_arm")["keyframes"]["0"]["POS_Z"] == -0.53
    assert timeline_for(exported, "right_arm")["keyframes"]["0"]["POS_Z"] == -0.43
    assert timeline_for(exported, "hat")["keyframes"] == {}


def test_export_rejects_empty_capture(tmp_path):
    template_path = tmp_path / "template.miobject"
    make_template(template_path)
    with pytest.raises(MiobjectError):
        export_miobject(template_path, tmp_path / "out.miobject", [], fps=30)


def _rotate_pose_y(pose, angle_degrees, indices):
    import math

    rotated = {
        "world": [list(point) for point in pose["world"]],
        "image": pose.get("image", []),
    }
    angle = math.radians(angle_degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    for index in indices:
        x, y, z, *extra = rotated["world"][index]
        rotated["world"][index] = [
            cosine * x + sine * z,
            y,
            -sine * x + cosine * z,
            *extra,
        ]
    return rotated


def test_torso_forward_bow_uses_3d_relative_rotation():
    neutral = standing_pose()
    bowed = standing_pose()
    for index in (11, 12):
        bowed["world"][index][2] -= 0.42

    controls = calculate_pose_controls(bowed, reference_pose=neutral)
    assert abs(controls["body"]["ROT_X"]) > 8.0
    assert abs(controls["body"]["ROT_Y"]) < 2.0
    assert abs(controls["body"]["ROT_Z"]) < 2.0


def test_torso_turn_uses_independent_yaw_without_fake_bow():
    neutral = standing_pose()
    rotated = _rotate_pose_y(neutral, 38.0, (11, 12, 23, 24))

    controls = calculate_pose_controls(rotated, reference_pose=neutral)
    assert abs(controls["body"]["ROT_Y"]) > 25.0
    assert abs(controls["body"]["ROT_X"]) < 2.0
    assert abs(controls["body"]["ROT_Z"]) < 2.0


def test_arms_stay_local_when_the_torso_turns():
    neutral = standing_pose()
    rotated = _rotate_pose_y(
        neutral, 42.0, (11, 12, 13, 14, 15, 16, 23, 24)
    )

    original = calculate_pose_controls(neutral, reference_pose=neutral)
    controls = calculate_pose_controls(rotated, reference_pose=neutral)
    for part in ("left_arm", "right_arm"):
        for field in ("ROT_X", "ROT_Z", "BEND_ANGLE_X"):
            assert abs(controls[part][field] - original[part][field]) < 1.5


def test_leg_motion_no_longer_applies_amplitude_compression():
    assert _shape_motion_delta("left_leg", "ROT_Z", 0.5) == 0.0
    assert _shape_motion_delta("left_leg", "ROT_Z", 20.0) == 19.0
    assert _shape_motion_delta("right_leg", "BEND_ANGLE_X", 20.0) == 18.5
    assert _shape_motion_delta("body", "ROT_X", 20.0) == 20.0


def test_robust_smoothing_rejects_single_frame_spike_but_keeps_motion():
    filtered = _smooth([0.0, 0.0, 35.0, 0.0, 0.0], True, fps=30)
    assert max(abs(value) for value in filtered) < 1.0
    gradual = _smooth([0.0, 2.0, 4.0, 6.0, 8.0], True, fps=30)
    assert gradual[-1] > 2.0



def _rotate_pose_x(pose, angle_degrees, indices):
    import math

    rotated = {
        "world": [list(point) for point in pose["world"]],
        "image": pose.get("image", []),
    }
    angle = math.radians(angle_degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    for index in indices:
        x, y, z, *extra = rotated["world"][index]
        rotated["world"][index] = [
            x,
            cosine * y - sine * z,
            sine * y + cosine * z,
            *extra,
        ]
    return rotated


def test_segment_rotation_reconstructs_a_diagonal_3d_direction():
    from math import cos, radians, sin, sqrt

    local_right = 0.6
    local_up = -0.3
    local_forward = sqrt(0.55)
    world_direction = (
        local_right,
        -local_up,
        -local_forward,
    )

    rot_x, rot_z = _segment_rotation(
        (0.0, 0.0, 0.0),
        world_direction,
        WORLD_REFERENCE_FRAME,
    )

    x = radians(rot_x)
    z = radians(rot_z)
    reconstructed = (
        sin(z) * cos(x),
        -cos(z) * cos(x),
        sin(x),
    )

    assert reconstructed[0] == pytest.approx(local_right, abs=1e-6)
    assert reconstructed[1] == pytest.approx(local_up, abs=1e-6)
    assert reconstructed[2] == pytest.approx(local_forward, abs=1e-6)
    assert -90.0 <= rot_x <= 90.0


def test_torso_frame_falls_back_to_hip_width_when_shoulders_are_degenerate():
    pose = standing_pose()
    # Keep the shoulder midpoint fixed but make its width axis parallel to up.
    pose["world"][11] = [0.0, -1.45, 0.0, 1.0, 1.0]
    pose["world"][12] = [0.0, -1.05, 0.0, 1.0, 1.0]

    frame = _torso_frame(pose)

    assert frame is not None
    assert frame[0] == pytest.approx((1.0, 0.0, 0.0), abs=1e-6)
    assert frame[1] == pytest.approx((0.0, -1.0, 0.0), abs=1e-6)
    assert frame[2] == pytest.approx((0.0, 0.0, -1.0), abs=1e-6)


def test_torso_rotation_is_not_artificially_clipped_at_89_degrees():
    neutral = standing_pose()
    rotated = _rotate_pose_x(neutral, 100.0, (11, 12, 23, 24))

    controls = calculate_pose_controls(rotated, reference_pose=neutral)

    assert controls["body"]["ROT_X"] == pytest.approx(100.0, abs=0.01)
    assert abs(controls["body"]["ROT_Y"]) < 0.01
    assert abs(controls["body"]["ROT_Z"]) < 0.01
