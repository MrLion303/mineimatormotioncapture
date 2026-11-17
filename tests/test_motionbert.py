import numpy as np
import pytest

from mocap_app.motionbert import (
    map_h36m_keypoints_to_pose,
    resample_capture_timestamps,
)


def test_motionbert_h36m_joints_map_to_expected_body_landmarks():
    joints = np.zeros((17, 3), dtype=np.float32)
    for index in range(17):
        joints[index] = (index * -0.1, index * 0.3, index * 0.2)

    pose = map_h36m_keypoints_to_pose(joints)
    world = pose["world"]

    assert len(world) == 33
    assert world[11][:3] == pytest.approx([1.1, -2.2, -3.3])
    assert world[13][:3] == pytest.approx([1.2, -2.4, -3.6])
    assert world[15][:3] == pytest.approx([1.3, -2.6, -3.9])
    assert world[12][:3] == pytest.approx([1.4, -2.8, -4.2])
    assert world[14][:3] == pytest.approx([1.5, -3.0, -4.5])
    assert world[16][:3] == pytest.approx([1.6, -3.2, -4.8])
    assert world[23][:3] == pytest.approx([0.4, -0.8, -1.2])
    assert world[24][:3] == pytest.approx([0.1, -0.2, -0.3])
    assert world[25][:3] == pytest.approx([0.5, -1.0, -1.5])
    assert world[26][:3] == pytest.approx([0.2, -0.4, -0.6])
    assert world[27][:3] == pytest.approx([0.6, -1.2, -1.8])
    assert world[28][:3] == pytest.approx([0.3, -0.6, -0.9])
    assert all(value == 1.0 for value in world[11][3:])


def test_motionbert_mapping_builds_head_helper_landmarks():
    joints = np.zeros((17, 3), dtype=np.float32)
    joints[9] = (0.0, 0.0, 0.0)
    joints[10] = (0.0, 0.0, 1.0)
    joints[11] = (0.5, 0.0, 0.8)
    joints[14] = (-0.5, 0.0, 0.8)

    pose = map_h36m_keypoints_to_pose(joints)
    world = pose["world"]

    assert world[0][:3] == pytest.approx([0.0, -1.3, 0.0])
    assert world[7][:3] == pytest.approx([-0.12, -1.0, 0.0])
    assert world[8][:3] == pytest.approx([0.12, -1.0, 0.0])


@pytest.mark.parametrize(
    "keypoints",
    [
        np.zeros((16, 3), dtype=np.float32),
        np.zeros((17, 2), dtype=np.float32),
        np.full((17, 3), np.nan, dtype=np.float32),
    ],
)
def test_motionbert_mapping_rejects_invalid_3d_output(keypoints):
    with pytest.raises(ValueError):
        map_h36m_keypoints_to_pose(keypoints)



def test_capture_resampling_converts_60_fps_to_uniform_30_fps():
    source_times = [index / 60.0 for index in range(5)]

    source_indices, target_times = resample_capture_timestamps(source_times)

    assert source_indices == [0, 2, 4]
    assert target_times == pytest.approx([0.0, 1 / 30, 2 / 30])


def test_capture_resampling_duplicates_frames_when_source_is_slower():
    source_times = [0.0, 1 / 15, 2 / 15]

    source_indices, target_times = resample_capture_timestamps(source_times)

    assert source_indices == [0, 0, 1, 1, 2]
    assert target_times == pytest.approx([index / 30 for index in range(5)])


def test_capture_resampling_rejects_non_monotonic_timestamps():
    with pytest.raises(ValueError, match="crecientes"):
        resample_capture_timestamps([0.0, 0.1, 0.05])



def test_motionbert_mapping_preserves_keypoint_confidence():
    joints = np.zeros((17, 3), dtype=np.float32)
    scores = np.ones(17, dtype=np.float32)
    scores[11] = 0.2
    scores[10] = 0.4

    pose = map_h36m_keypoints_to_pose(joints, scores)
    world = pose["world"]

    assert world[11][3:] == pytest.approx([0.2, 0.2])
    assert world[0][3:] == pytest.approx([0.4, 0.4])
    assert world[7][3:] == pytest.approx([0.2, 0.2])


@pytest.mark.parametrize(
    "scores",
    [
        np.zeros(16, dtype=np.float32),
        np.full(17, np.nan, dtype=np.float32),
    ],
)
def test_motionbert_mapping_rejects_invalid_confidence(scores):
    joints = np.zeros((17, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="puntuaciones"):
        map_h36m_keypoints_to_pose(joints, scores)
