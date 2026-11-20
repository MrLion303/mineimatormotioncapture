import json

import numpy as np
import pytest

from mocap_app.miobject import (
    MiobjectError,
    _select_keyframes,
    _shape_motion_delta,
    _smooth,
    _serialize_rotation_axes,
    export_miobject,
    IK_HELPER_CONFIG,
)
from mocap_app.retarget import (
    WORLD_REFERENCE_FRAME,
    _segment_rotation,
    _torso_frame,
    _matrix_to_yxz,
    calculate_pose_controls,
    calculate_pose_sequence,
    calculate_ik_target_sequence,
)


def standing_pose():
    world = [[0.0, 0.0, 0.0, 1.0, 1.0] for _ in range(33)]
    points = {
        0: (0.0, -1.9, -0.1),
        7: (-0.12, -1.75, 0.0),
        8: (0.12, -1.75, 0.0),
        11: (-0.25, -1.25, 0.0),
        12: (0.25, -1.25, 0.0),
        13: (-0.25, -0.85, 0.0),
        14: (0.25, -0.85, 0.0),
        15: (-0.25, -0.45, 0.0),
        16: (0.25, -0.45, 0.0),
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


def test_supplied_working_torso_turn_channel_reference():
    from pathlib import Path

    reference_path = (
        Path(__file__).parent
        / "fixtures"
        / "references"
        / "torso_turn_channel_reference.json"
    )
    reference = json.loads(reference_path.read_text(encoding="utf-8"))

    assert reference["format"] == 35
    assert reference["body"]["parent"] == "char"
    assert reference["body"]["keyframes"]["15"]["BEND_ANGLE_Z"] == pytest.approx(61.41356)
    assert reference["body"]["keyframes"]["45"]["BEND_ANGLE_Z"] == pytest.approx(-47.32366)
    assert reference["children"] == ["head", "left_arm", "right_arm"]



def test_export_preserves_non_neutral_first_frame_pose(tmp_path):
    template_path = tmp_path / "template.miobject"
    output_path = tmp_path / "captured.miobject"
    make_template(template_path)

    first_pose = standing_pose()
    # Start with the left arm raised and bent, as in the supplied Rickroll.
    first_pose["world"][13] = [-0.03, -1.55, 0.0, 1.0, 1.0]
    first_pose["world"][15] = [0.28, -1.62, -0.02, 1.0, 1.0]
    # Start with the right knee flexed instead of treating the stance as rest.
    first_pose["world"][26] = [0.34, 0.68, -0.12, 1.0, 1.0]
    first_pose["world"][28] = [0.52, 1.25, -0.20, 1.0, 1.0]

    expected = calculate_pose_controls(first_pose, reference_pose=first_pose)
    assert abs(expected["left_arm"]["ROT_Z"]) > 10.0
    assert expected["left_arm"]["BEND_ANGLE_X"] > 10.0
    assert abs(expected["right_leg"]["ROT_Z"]) > 1.0

    export_miobject(
        template_path,
        output_path,
        [(0.0, first_pose), (0.5, first_pose)],
        fps=30,
        duration_seconds=0.5,
        optimize_keyframes=False,
        smooth_motion=False,
    )

    exported = json.loads(output_path.read_text(encoding="utf-8"))
    left_arm = timeline_for(exported, "left_arm")["keyframes"]["0"]
    right_leg = timeline_for(exported, "right_leg")["keyframes"]["0"]

    assert left_arm["ROT_Y"] == pytest.approx(expected["left_arm"]["ROT_Z"], abs=0.01)
    assert left_arm["ROT_Z"] == pytest.approx(expected["left_arm"]["ROT_Y"], abs=0.01)
    assert left_arm["BEND_ANGLE_X"] == pytest.approx(
        expected["left_arm"]["BEND_ANGLE_X"], abs=0.01
    )
    expected_leg_y = _shape_motion_delta(
        "right_leg", "ROT_Y", expected["right_leg"]["ROT_Z"]
    )
    expected_leg_z = _shape_motion_delta(
        "right_leg", "ROT_Z", expected["right_leg"]["ROT_Y"]
    )
    assert right_leg["ROT_Y"] == pytest.approx(expected_leg_y, abs=0.01)
    assert right_leg["ROT_Z"] == pytest.approx(expected_leg_z, abs=0.01)
    body = timeline_for(exported, "body")["keyframes"]["0"]
    assert all(abs(body[field]) < 0.001 for field in (
        "BEND_ANGLE_X", "BEND_ANGLE_Y", "BEND_ANGLE_Z"
    ))


def test_direct_retarget_uses_the_native_channels_from_the_mineimator_example():
    controls = calculate_pose_controls(standing_pose())

    assert set(controls) == {
        "body", "head", "left_arm", "right_arm", "left_leg", "right_leg"
    }
    assert set(controls["body"]) == {
        "BEND_ANGLE_X", "BEND_ANGLE_Y", "BEND_ANGLE_Z"
    }
    assert set(controls["head"]) == {"ROT_X", "ROT_Y", "ROT_Z"}

    for part in ("left_arm", "right_arm", "left_leg", "right_leg"):
        assert set(controls[part]) == {
            "ROT_X", "ROT_Y", "ROT_Z", "BEND_ANGLE_X"
        }
        assert all(np.isfinite(controls[part][field]) for field in ("ROT_X", "ROT_Y", "ROT_Z"))

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
        rotation_delta = sum(
            abs(controls[part][field] - original[part][field])
            for field in ("ROT_X", "ROT_Y", "ROT_Z")
        )
        assert rotation_delta > 5.0
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

    original = calculate_pose_controls(neutral, reference_pose=neutral)
    controls = calculate_pose_controls(moved, reference_pose=neutral)
    assert any(
        abs(controls["body"][field] - original["body"][field]) > 1.0
        for field in ("BEND_ANGLE_X", "BEND_ANGLE_Y", "BEND_ANGLE_Z")
    )
    head_moved = standing_pose()
    head_moved["world"][0][0] += 0.18
    head_moved["world"][7][1] += 0.05
    head_moved["world"][8][1] -= 0.05
    head_controls = calculate_pose_controls(head_moved, reference_pose=neutral)
    assert any(
        abs(head_controls["head"][field] - original["head"][field]) > 1.0
        for field in ("ROT_X", "ROT_Y", "ROT_Z")
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
        "BEND_ANGLE_X", "BEND_ANGLE_Y", "BEND_ANGLE_Z"
    ))
    # Template rotation values stay fixed; the body uses native bend channels.
    for field in ("ROT_X", "ROT_Y", "ROT_Z"):
        assert len({frame[field] for frame in body["keyframes"].values()}) == 1

    head = timeline_for(exported, "head")
    first_head = head["keyframes"]["0"]
    assert abs(first_head["ROT_X"]) < 0.001
    assert abs(first_head["ROT_Y"]) < 0.001
    assert abs(first_head["ROT_Z"]) < 0.001
    assert any(
        abs(frame["ROT_Y"]) + abs(frame["ROT_Z"]) > 1.0
        for frame in head["keyframes"].values()
    )

    for part in ("left_arm", "right_arm", "left_leg", "right_leg"):
        first = timeline_for(exported, part)["keyframes"]["0"]
        assert abs(first["ROT_X"]) < 0.001
        assert abs(first["ROT_Z"]) < 0.001
        assert abs(first["BEND_ANGLE_X"]) < 1.0

    left_leg = timeline_for(exported, "left_leg")
    right_leg = timeline_for(exported, "right_leg")
    assert any(
        abs(frame["ROT_X"]) + abs(frame["ROT_Y"]) + abs(frame["ROT_Z"]) > 1.0
        for frame in left_leg["keyframes"].values()
    )
    assert any(
        abs(frame["ROT_X"]) + abs(frame["ROT_Y"]) + abs(frame["ROT_Z"]) > 1.0
        for frame in right_leg["keyframes"].values()
    )

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
    assert abs(controls["body"]["BEND_ANGLE_X"]) > 8.0
    assert abs(controls["body"]["BEND_ANGLE_Y"]) < 2.0
    assert abs(controls["body"]["BEND_ANGLE_Z"]) < 2.0


def test_torso_turn_uses_independent_yaw_without_fake_bow():
    neutral = standing_pose()
    rotated = _rotate_pose_y(neutral, 38.0, (11, 12, 23, 24))

    controls = calculate_pose_controls(rotated, reference_pose=neutral)
    assert abs(controls["body"]["BEND_ANGLE_Z"]) > 25.0
    assert abs(controls["body"]["BEND_ANGLE_X"]) < 2.0
    assert abs(controls["body"]["BEND_ANGLE_Y"]) < 2.0


def test_torso_side_lean_uses_roll_without_fake_yaw_or_forward_bow():
    neutral = standing_pose()
    tilted = _rotate_pose_z(neutral, 32.0, (11, 12, 23, 24))

    controls = calculate_pose_controls(tilted, reference_pose=neutral)
    assert abs(controls["body"]["BEND_ANGLE_Y"]) > 20.0
    assert abs(controls["body"]["BEND_ANGLE_X"]) < 2.0
    assert abs(controls["body"]["BEND_ANGLE_Z"]) < 2.0


def test_arms_stay_local_when_the_torso_turns():
    neutral = standing_pose()
    rotated = _rotate_pose_y(
        neutral, 42.0, (11, 12, 13, 14, 15, 16, 23, 24)
    )

    original = calculate_pose_controls(neutral, reference_pose=neutral)
    controls = calculate_pose_controls(rotated, reference_pose=neutral)
    for part in ("left_arm", "right_arm"):
        for field in ("ROT_X", "ROT_Y", "ROT_Z", "BEND_ANGLE_X"):
            assert abs(controls[part][field] - original[part][field]) < 1.5


def test_leg_motion_keeps_the_captured_amplitude():
    assert _shape_motion_delta("left_leg", "ROT_Z", 0.5) == pytest.approx(0.5)
    assert _shape_motion_delta("left_leg", "ROT_Z", 20.0) == pytest.approx(20.0)
    assert _shape_motion_delta("right_leg", "BEND_ANGLE_X", 20.0) == pytest.approx(20.0)
    assert _shape_motion_delta("body", "BEND_ANGLE_X", 20.0) == pytest.approx(20.0)


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


def _rotate_pose_z(pose, angle_degrees, indices):
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
            cosine * x - sine * y,
            sine * x + cosine * y,
            z,
            *extra,
        ]
    return rotated


def test_torso_lateral_roll_uses_bend_y_not_yaw_bend_z():
    neutral = standing_pose()
    tilted = _rotate_pose_z(neutral, 32.0, (11, 12, 23, 24))

    controls = calculate_pose_controls(tilted, reference_pose=neutral)
    assert abs(controls["body"]["BEND_ANGLE_Y"]) > 20.0
    assert abs(controls["body"]["BEND_ANGLE_X"]) < 2.0
    assert abs(controls["body"]["BEND_ANGLE_Z"]) < 2.0


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


def test_torso_frame_keeps_forward_axis_aligned_with_face_when_shoulders_are_mirrored():
    pose = standing_pose()
    # MediaPipe's left shoulder/hip lie on image-right for a person facing camera.
    # Their anatomical right-minus-left width points toward the opposite side.
    for left_index, right_index in ((11, 12), (23, 24)):
        pose["world"][left_index], pose["world"][right_index] = (
            pose["world"][right_index],
            pose["world"][left_index],
        )

    frame = _torso_frame(pose)
    face_direction = (
        pose["world"][0][0] - (pose["world"][7][0] + pose["world"][8][0]) / 2.0,
        pose["world"][0][1] - (pose["world"][7][1] + pose["world"][8][1]) / 2.0,
        pose["world"][0][2] - (pose["world"][7][2] + pose["world"][8][2]) / 2.0,
    )

    assert frame is not None
    assert sum(frame[2][index] * face_direction[index] for index in range(3)) > 0.0
    assert frame[0] == pytest.approx((1.0, 0.0, 0.0), abs=1e-6)
    assert frame[2] == pytest.approx((0.0, 0.0, -1.0), abs=1e-6)


def test_torso_rotation_is_not_artificially_clipped_at_89_degrees():
    neutral = standing_pose()
    rotated = _rotate_pose_x(neutral, 100.0, (11, 12, 23, 24))

    controls = calculate_pose_controls(rotated, reference_pose=neutral)

    assert controls["body"]["BEND_ANGLE_X"] == pytest.approx(100.0, abs=0.01)
    assert abs(controls["body"]["BEND_ANGLE_Y"]) < 0.01
    assert abs(controls["body"]["BEND_ANGLE_Z"]) < 0.01



def test_yxz_matrix_conversion_reconstructs_full_3d_orientation():
    angles_yxz = (31.0, -22.0, 47.0)
    radians = np.radians(angles_yxz)
    cy, cx, cz = np.cos(radians)
    sy, sx, sz = np.sin(radians)
    ry = np.asarray(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)), dtype=float)
    rx = np.asarray(((1, 0, 0), (0, cx, -sx), (0, sx, cx)), dtype=float)
    rz = np.asarray(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)), dtype=float)
    matrix = ry @ rx @ rz

    converted = _matrix_to_yxz(matrix)
    assert converted == pytest.approx(angles_yxz, abs=1e-6)
    rebuilt_y, rebuilt_x, rebuilt_z = np.radians(converted)
    cy, cx, cz = np.cos((rebuilt_y, rebuilt_x, rebuilt_z))
    sy, sx, sz = np.sin((rebuilt_y, rebuilt_x, rebuilt_z))
    rebuilt = np.asarray(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy))) @ np.asarray(
        ((1, 0, 0), (0, cx, -sx), (0, sx, cx))
    ) @ np.asarray(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)))
    assert rebuilt == pytest.approx(matrix, abs=1e-6)


def test_mineimator_rotation_file_fields_swap_y_and_z_only_when_serialized():
    semantic = {"ROT_X": 12.5, "ROT_Y": -27.0, "ROT_Z": 43.0, "BEND_ANGLE_X": 18.0}
    raw = _serialize_rotation_axes(semantic)

    assert raw["ROT_X"] == pytest.approx(12.5)
    assert raw["ROT_Y"] == pytest.approx(43.0)
    assert raw["ROT_Z"] == pytest.approx(-27.0)
    assert raw["BEND_ANGLE_X"] == pytest.approx(18.0)
    assert semantic["ROT_Y"] == pytest.approx(-27.0)
    assert semantic["ROT_Z"] == pytest.approx(43.0)


def test_retarget_uses_observable_swing_axes_without_inventing_axial_twist():
    pose = standing_pose()
    pose["world"][13] = [-0.12, -0.93, 0.18, 1.0, 1.0]
    pose["world"][15] = [0.10, -0.62, -0.16, 1.0, 1.0]

    controls = calculate_pose_controls(pose, reference_pose=standing_pose())
    arm = controls["left_arm"]

    assert set(arm) == {"ROT_X", "ROT_Y", "ROT_Z", "BEND_ANGLE_X"}
    assert abs(arm["ROT_X"]) > 2.0
    assert abs(arm["ROT_Z"]) > 2.0
    assert arm["ROT_Y"] == pytest.approx(0.0, abs=1e-8)
    assert 0.0 <= arm["BEND_ANGLE_X"] <= 150.0


def test_pose_sequence_keeps_limb_frame_continuity_through_straight_pose():
    poses = []
    for offset in (-0.06, -0.025, 0.0, 0.025, 0.06):
        pose = standing_pose()
        pose["world"][13] = [-0.25, -0.85, 0.0, 1.0, 1.0]
        pose["world"][15] = [-0.25 + offset, -0.45, 0.0, 1.0, 1.0]
        poses.append((len(poses) / 30.0, pose))

    sequence = calculate_pose_sequence(poses, reference_pose=poses[0][1])
    rotations = [
        np.asarray([controls["left_arm"][field] for field in ("ROT_X", "ROT_Y", "ROT_Z")])
        for _, controls in sequence
    ]
    assert len(rotations) == len(poses)
    assert max(
        float(np.max(np.abs(right - left)))
        for left, right in zip(rotations, rotations[1:])
    ) < 120.0



def test_yxz_euler_angles_stay_continuous_through_gimbal_singularity():
    from math import cos, radians, sin

    def rotation_yxz(y, x, z):
        y, x, z = (radians(value) for value in (y, x, z))
        cy, cx, cz = cos(y), cos(x), cos(z)
        sy, sx, sz = sin(y), sin(x), sin(z)
        ry = np.asarray(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)), dtype=float)
        rx = np.asarray(((1, 0, 0), (0, cx, -sx), (0, sx, cx)), dtype=float)
        rz = np.asarray(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)), dtype=float)
        return ry @ rx @ rz

    previous = None
    sequence = []
    for middle_angle in range(80, 101, 2):
        matrix = rotation_yxz(25.0, float(middle_angle), -10.0)
        angles = _matrix_to_yxz(matrix, previous)
        sequence.append(angles)
        previous = angles

    assert max(
        max(abs(right[axis] - left[axis]) for axis in range(3))
        for left, right in zip(sequence, sequence[1:])
    ) < 15.0
    assert sequence[-1][1] == pytest.approx(100.0, abs=1e-6)


def test_yxz_euler_alternate_branch_does_not_accumulate_hundreds_of_degrees():
    from math import cos, radians, sin

    def rotation_y(angle):
        value = radians(angle)
        return np.asarray(
            ((cos(value), 0, sin(value)), (0, 1, 0), (-sin(value), 0, cos(value))),
            dtype=float,
        )

    previous = None
    sequence = []
    for angle in (160, 170, 179, 181, 190, 200, 215, 230):
        result = _matrix_to_yxz(rotation_y(float(angle)), previous)
        sequence.append(result)
        previous = result

    assert max(abs(value) for row in sequence for value in row) < 360.0
    assert max(
        abs(right[0] - left[0])
        for left, right in zip(sequence, sequence[1:])
    ) < 20.0



def test_export_duration_matches_mineimator_24_fps_project(tmp_path):
    template_path = tmp_path / "template.miobject"
    output_path = tmp_path / "captured.miobject"
    make_template(template_path)

    source_duration = 30.1166666667
    result = export_miobject(
        template_path,
        output_path,
        [(0.0, standing_pose()), (source_duration, standing_pose())],
        fps=24,
        duration_seconds=source_duration,
        optimize_keyframes=False,
        smooth_motion=False,
    )

    assert result["fps"] == 24
    assert result["final_frame"] == 723
    assert result["final_frame"] / 24 == pytest.approx(30.125)
    assert abs(result["final_frame"] / 24 - source_duration) < 1 / 24



def test_forearm_orbit_does_not_create_fictitious_upper_arm_twist():
    import math

    base = standing_pose()
    shoulder = np.asarray(base["world"][11][:3], dtype=float)
    elbow = np.asarray(base["world"][13][:3], dtype=float)
    axis = shoulder - elbow
    axis /= np.linalg.norm(axis)

    # Keep the upper arm fixed and rotate a bent forearm around its own
    # proximal axis. The wrist path must not become a full rotation of the
    # upper arm in the exported controls.
    lower = np.asarray((0.25, 0.25, 0.0), dtype=float)
    samples = []
    for step, angle_degrees in enumerate(range(0, 721, 30)):
        angle = math.radians(angle_degrees)
        rotated_lower = (
            lower * math.cos(angle)
            + np.cross(axis, lower) * math.sin(angle)
            + axis * np.dot(axis, lower) * (1.0 - math.cos(angle))
        )
        pose = {
            "world": [list(point) for point in base["world"]],
            "image": [],
        }
        wrist = elbow + rotated_lower
        pose["world"][15] = [*wrist.tolist(), 1.0, 1.0]
        samples.append((step / 24.0, pose))

    converted = calculate_pose_sequence(samples, reference_pose=samples[0][1])
    rotations = [
        controls["left_arm"]
        for _, controls in converted
        if "left_arm" in controls
    ]
    assert len(rotations) == len(samples)
    for field in ("ROT_X", "ROT_Y", "ROT_Z", "BEND_ANGLE_X"):
        values = [frame[field] for frame in rotations]
        assert max(values) - min(values) < 0.1



def test_root_translation_preserves_lateral_step_vertical_bounce_and_depth():
    reference = standing_pose()
    moved = standing_pose()
    for point in moved["world"]:
        point[0] += 0.20
        point[1] -= 0.10
        point[2] += 0.30

    sequence = calculate_pose_sequence(
        [(0.0, reference), (1.0, moved)],
        reference_pose=reference,
    )
    root = sequence[-1][1]["root"]

    assert root["POS_X"] > 0.08
    assert root["POS_Y"] > 0.03
    assert root["POS_Z"] > 0.12


def test_legs_are_retargeted_in_character_root_space_not_torso_yaw():
    neutral = standing_pose()
    # Twist only the upper torso around the pelvis. The hip-to-knee and
    # knee-to-ankle vectors stay fixed in character-root space.
    torso_turn = _rotate_pose_y(neutral, 42.0, (11, 12))

    sequence = calculate_pose_sequence(
        [(0.0, neutral), (1.0, torso_turn)],
        reference_pose=neutral,
    )
    first = sequence[0][1]
    turned = sequence[1][1]

    for part in ("left_leg", "right_leg"):
        for field in ("ROT_X", "ROT_Y", "ROT_Z", "BEND_ANGLE_X"):
            assert turned[part][field] == pytest.approx(first[part][field], abs=1e-5)


def test_export_writes_root_translation_without_changing_template_origin(tmp_path):
    template_path = tmp_path / "template.miobject"
    output_path = tmp_path / "captured.miobject"
    make_template(template_path)
    reference = standing_pose()
    moved = standing_pose()
    for point in moved["world"]:
        point[0] += 0.20
        point[1] -= 0.10

    export_miobject(
        template_path,
        output_path,
        [(0.0, reference), (1.0, moved)],
        fps=24,
        duration_seconds=1.0,
        optimize_keyframes=False,
        smooth_motion=False,
    )
    exported = json.loads(output_path.read_text(encoding="utf-8"))
    character = next(item for item in exported["timelines"] if item["type"] == "char")

    assert character["keyframes"]["0"]["POS_X"] == pytest.approx(12.0)
    assert character["keyframes"]["0"]["POS_Y"] == pytest.approx(24.0)
    assert character["keyframes"]["24"]["POS_X"] > 12.08
    assert character["keyframes"]["24"]["POS_Y"] > 24.03



def _with_image_landmarks(pose, updates, image_size=(1000, 1000)):
    pose = {
        "world": [list(point) for point in pose["world"]],
        "image": [[0.5, 0.5, 0.0, 1.0, 1.0] for _ in range(33)],
        "image_size": image_size,
    }
    for index, (x, y) in updates.items():
        pose["image"][index] = [x, y, 0.0, 1.0, 1.0]
    return pose


def test_leg_knee_uses_screen_geometry_instead_of_noisy_world_depth():
    pose = standing_pose()
    image = {
        0: (0.5, 0.1),
        11: (0.42, 0.3), 12: (0.58, 0.3),
        23: (0.4, 0.45), 25: (0.43, 0.65), 27: (0.46, 0.85),
        24: (0.6, 0.45), 26: (0.6, 0.65), 28: (0.6, 0.85),
    }
    pose = _with_image_landmarks(pose, image)

    # MediaPipe world depth suggests a deep bend, despite the visible 2D leg
    # remaining almost straight. The output must trust the image plane.
    pose["world"][25][2] = 0.35
    pose["world"][27][2] = -0.25

    controls = calculate_pose_controls(pose, reference_pose=pose)
    left_leg = controls["left_leg"]

    assert left_leg["BEND_ANGLE_X"] < 8.0
    assert left_leg["ROT_Z"] == pytest.approx(
        np.degrees(np.arctan2(0.03, 0.20)), abs=0.5
    )
    assert abs(left_leg["ROT_X"]) <= 18.0


def test_root_tracks_image_space_and_anchors_to_the_supporting_foot():
    base = standing_pose()
    reference = _with_image_landmarks(base, {
        0: (0.5, 0.1),
        23: (0.4, 0.5), 24: (0.6, 0.5),
        27: (0.4, 0.9), 28: (0.6, 0.88),
    })
    moved = _with_image_landmarks(base, {
        0: (0.55, 0.07),
        23: (0.45, 0.47), 24: (0.65, 0.47),
        27: (0.45, 0.87), 28: (0.65, 0.85),
    })

    sequence = calculate_pose_sequence(
        [(0.0, reference), (1.0, moved)],
        reference_pose=reference,
    )
    root = sequence[-1][1]["root"]

    assert root["POS_X"] == pytest.approx(0.1125, abs=0.01)
    assert root["POS_Y"] == pytest.approx(0.0675, abs=0.01)
    assert root["POS_Z"] == pytest.approx(0.0, abs=1e-8)


def test_supporting_foot_prevents_root_lift_when_only_one_ankle_moves():
    base = standing_pose()
    reference = _with_image_landmarks(base, {
        0: (0.5, 0.1),
        23: (0.4, 0.5), 24: (0.6, 0.5),
        27: (0.4, 0.9), 28: (0.6, 0.9),
    })
    one_foot_up = _with_image_landmarks(base, {
        0: (0.5, 0.1),
        23: (0.4, 0.5), 24: (0.6, 0.5),
        27: (0.4, 0.8), 28: (0.6, 0.9),
    })

    sequence = calculate_pose_sequence(
        [(0.0, reference), (1.0, one_foot_up)],
        reference_pose=reference,
    )

    assert sequence[-1][1]["root"]["POS_Y"] == pytest.approx(0.0, abs=1e-8)



def _ik_pose(wrist_left=(0.48, 0.40), ankle_left=(0.44, 0.90)):
    base = standing_pose()
    return _with_image_landmarks(base, {
        0: (0.5, 0.1),
        11: (0.42, 0.3), 12: (0.58, 0.3),
        13: (0.35, 0.42), 15: wrist_left,
        14: (0.65, 0.42), 16: (0.52, 0.40),
        23: (0.44, 0.5), 24: (0.56, 0.5),
        25: (0.45, 0.68), 27: ankle_left,
        26: (0.55, 0.68), 28: (0.56, 0.90),
    })


def test_ik_targets_preserve_rig_segment_lengths_and_follow_the_wrist():
    first = _ik_pose()
    second = _ik_pose(wrist_left=(0.50, 0.35))

    sequence = calculate_ik_target_sequence(
        [(0.0, first), (0.5, second)],
        reference_pose=first,
    )
    first_left_arm = sequence[0][1]["left_arm"]
    second_left_arm = sequence[1][1]["left_arm"]

    start = np.asarray(first_left_arm["start"])
    pole = np.asarray(first_left_arm["pole"])
    target = np.asarray(first_left_arm["target"])

    assert np.linalg.norm(pole - start) == pytest.approx(4.0, abs=1e-5)
    assert np.linalg.norm(target - pole) == pytest.approx(6.0, abs=1e-5)
    assert np.linalg.norm(
        np.asarray(second_left_arm["target"]) - target
    ) > 0.5
    assert np.asarray(second_left_arm["pole"]) == pytest.approx(pole, abs=1e-5)


def test_ik_leg_targets_use_screen_knee_geometry_and_12_unit_chain():
    pose = _ik_pose(ankle_left=(0.44, 0.90))
    bent = _ik_pose(ankle_left=(0.56, 0.76))

    sequence = calculate_ik_target_sequence(
        [(0.0, pose), (0.5, bent)],
        reference_pose=pose,
    )
    leg = sequence[0][1]["left_leg"]
    start = np.asarray(leg["start"])
    pole = np.asarray(leg["pole"])
    target = np.asarray(leg["target"])

    assert np.linalg.norm(pole - start) == pytest.approx(6.0, abs=1e-5)
    assert np.linalg.norm(target - pole) == pytest.approx(6.0, abs=1e-5)
    assert np.linalg.norm(
        np.asarray(sequence[1][1]["left_leg"]["target"]) - target
    ) > 0.5


def test_export_connects_all_four_limbs_to_native_ik_targets(tmp_path):
    template_path = tmp_path / "template.miobject"
    output_path = tmp_path / "captured.miobject"
    make_template(template_path)

    first = _ik_pose()
    second = _ik_pose(wrist_left=(0.50, 0.35))
    export_miobject(
        template_path,
        output_path,
        [(0.0, first), (0.5, second)],
        fps=24,
        duration_seconds=0.5,
        optimize_keyframes=False,
        smooth_motion=False,
    )

    exported = json.loads(output_path.read_text(encoding="utf-8"))
    cameras = {
        timeline["id"]: timeline
        for timeline in exported["timelines"]
        if timeline.get("type") == "camera"
    }
    character = next(
        timeline for timeline in exported["timelines"]
        if timeline.get("type") == "char"
    )

    assert len(cameras) == 8
    for part, helper in IK_HELPER_CONFIG.items():
        limb = timeline_for(exported, part)
        first_key = limb["keyframes"]["0"]
        assert first_key["IK_TARGET"] == helper["target_id"]
        assert first_key["IK_TARGET_ANGLE"] == helper["pole_id"]
        assert first_key["IK_BLEND"] == pytest.approx(1.0)
        assert first_key["IK_ANGLE_OFFSET"] == pytest.approx(0.0)

        target_timeline = cameras[helper["target_id"]]
        pole_timeline = cameras[helper["pole_id"]]
        assert target_timeline["parent"] == character["id"]
        assert pole_timeline["parent"] == character["id"]
        assert target_timeline["hide"] is True
        assert pole_timeline["hide"] is True
        if part == "left_arm":
            assert (
                target_timeline["keyframes"]["0"]
                != target_timeline["keyframes"]["12"]
            )
        else:
            assert target_timeline["keyframes"]

    assert export_miobject.__name__ == "export_miobject"
