import numpy as np
import pytest

from mocap_app.motionbert import (
    _coco_scores_to_h36m,
    _mmpose_metadata_directory,
    _disable_motionbert_flip_test,
    _motionbert_inferencer_kwargs,
    _current_pose2d_image_landmarks,
    map_h36m_keypoints_to_pose,
    process_motionbert_video,
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


def test_motionbert_mapping_preserves_real_neck_to_head_axis_without_fake_face_points():
    joints = np.zeros((17, 3), dtype=np.float32)
    joints[9] = (0.0, 0.0, 0.0)
    joints[10] = (0.0, 0.0, 1.0)
    joints[11] = (0.5, 0.0, 0.8)
    joints[14] = (-0.5, 0.0, 0.8)

    pose = map_h36m_keypoints_to_pose(joints)

    assert pose["head_axis"]["neck"] == pytest.approx([0.0, 0.0, 0.0])
    assert pose["head_axis"]["head"] == pytest.approx([0.0, -1.0, 0.0])
    assert pose["head_axis"]["confidence"] == pytest.approx(1.0)
    # H36M has no nose or ears; those MediaPipe indices must stay unavailable.
    for index in (0, 7, 8):
        assert pose["world"][index][3:] == [0.0, 0.0]


def test_current_pose2d_image_landmarks_reads_rtpose_pixel_coordinates():
    from types import SimpleNamespace

    points = np.arange(34, dtype=np.float32).reshape(1, 17, 2)
    scores = np.linspace(0.0, 1.0, 17, dtype=np.float32).reshape(1, 17)
    sample = SimpleNamespace(pred_instances=SimpleNamespace(
        keypoints=points,
        keypoint_scores=scores,
    ))
    inferencer = SimpleNamespace(inferencer=SimpleNamespace(
        pose2d_model=SimpleNamespace(model=SimpleNamespace(
            dataset_meta={"dataset_name": "coco"},
        )),
        _buffer={"pose_est_results_list": [[sample]]},
    ))

    landmarks = _current_pose2d_image_landmarks(inferencer)

    assert len(landmarks) == 17
    assert landmarks[0] == pytest.approx([0.0, 1.0, 0.0])
    assert landmarks[-1] == pytest.approx([32.0, 33.0, 1.0])


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
    assert world[0][3:] == pytest.approx([0.0, 0.0])
    assert world[7][3:] == pytest.approx([0.0, 0.0])
    assert pose["head_axis"]["confidence"] == pytest.approx(0.4)


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

def test_windows_inferencer_skips_the_expensive_person_detector():
    kwargs = _motionbert_inferencer_kwargs("Windows")

    assert kwargs == {
        "pose3d": "human3d",
        "device": "cpu",
        "det_model": "whole_image",
    }


def test_non_windows_inferencer_keeps_the_human3d_default_detector():
    kwargs = _motionbert_inferencer_kwargs("Linux")

    assert kwargs == {"pose3d": "human3d", "device": "cpu"}


def test_motionbert_video_reuses_inferencer_without_leaking_sequence_state():
    from types import SimpleNamespace

    joints = np.zeros((17, 3), dtype=np.float32)
    joints[10] = (0.0, 0.0, 1.0)
    joints[11] = (0.5, 0.0, 0.8)
    joints[14] = (-0.5, 0.0, 0.8)
    scores = np.ones(17, dtype=np.float32)

    class FakeInferencer:
        def __init__(self):
            self.inferencer = SimpleNamespace(
                _buffer={"previous_recording": [object()]},
                _video_input=True,
            )

        def __call__(self, video_path):
            assert video_path.endswith("capture.avi")
            assert self.inferencer._buffer == {}
            assert not self.inferencer._video_input
            self.inferencer._buffer["current_recording"] = [object()]
            yield {
                "predictions": [[{
                    "keypoints": joints,
                    "keypoint_scores": scores,
                }]]
            }

    fake = FakeInferencer()
    emitted = []
    count = process_motionbert_video(
        "capture.avi",
        [12.5],
        lambda pose, timestamp: emitted.append((pose, timestamp)),
        lambda _message: None,
        inferencer=fake,
    )

    assert count == 1
    assert len(emitted) == 1
    assert emitted[0][1] == 12.5
    assert len(emitted[0][0]["world"]) == 33
    assert fake.inferencer._buffer == {}
    assert not fake.inferencer._video_input


def test_motionbert_video_rejects_non_increasing_timestamps_before_inference():
    class ShouldNotRun:
        def __call__(self, _video_path):
            raise AssertionError("The inferencer must not be called.")

    with pytest.raises(ValueError, match="crecientes"):
        process_motionbert_video(
            "capture.avi",
            [1.0, 1.0],
            lambda _pose, _timestamp: None,
            lambda _message: None,
            inferencer=ShouldNotRun(),
        )


def test_motionbert_video_rejects_more_results_than_timestamped_frames():
    class TooManyFrames:
        def __call__(self, _video_path):
            yield {"predictions": []}
            yield {"predictions": []}

    with pytest.raises(RuntimeError, match="más fotogramas"):
        process_motionbert_video(
            "capture.avi",
            [1.0],
            lambda _pose, _timestamp: None,
            lambda _message: None,
            inferencer=TooManyFrames(),
        )

def test_rtmpose_coco_confidence_is_converted_to_h36m_order():
    coco_scores = np.arange(17, dtype=np.float32) / 16.0

    h36m_scores = _coco_scores_to_h36m(coco_scores)

    assert h36m_scores.shape == (17,)
    assert h36m_scores[0] == pytest.approx(min(coco_scores[11], coco_scores[12]))
    assert h36m_scores[1:7] == pytest.approx(coco_scores[[12, 14, 16, 11, 13, 15]])
    assert h36m_scores[8] == pytest.approx(min(coco_scores[5], coco_scores[6]))
    assert h36m_scores[7] == pytest.approx(min(h36m_scores[0], h36m_scores[8]))
    assert h36m_scores[9] == pytest.approx(coco_scores[0])
    assert h36m_scores[10] == pytest.approx(min(coco_scores[1], coco_scores[2]))
    assert h36m_scores[11:17] == pytest.approx(coco_scores[[5, 7, 9, 6, 8, 10]])


def test_motionbert_prefers_real_rtmpose_scores_over_placeholder_3d_scores():
    from types import SimpleNamespace

    joints = np.zeros((17, 3), dtype=np.float32)
    joints[10] = (0.0, 0.0, 1.0)
    joints[11] = (0.5, 0.0, 0.8)
    joints[14] = (-0.5, 0.0, 0.8)
    coco_scores = np.arange(17, dtype=np.float32) / 16.0

    class FakeInferencer:
        def __init__(self):
            self.inferencer = SimpleNamespace(
                _buffer={"old": [object()]},
                _video_input=True,
                pose2d_model=SimpleNamespace(
                    model=SimpleNamespace(dataset_meta={"dataset_name": "coco"})
                ),
            )

        def __call__(self, _video_path):
            self.inferencer._buffer["pose_est_results_list"] = [[
                SimpleNamespace(
                    pred_instances=SimpleNamespace(keypoint_scores=coco_scores)
                )
            ]]
            yield {
                "predictions": [[{
                    "keypoints": joints,
                    "keypoint_scores": np.ones(17, dtype=np.float32),
                }]]
            }

    emitted = []
    process_motionbert_video(
        "capture.avi",
        [7.0],
        lambda pose, timestamp: emitted.append((pose, timestamp)),
        lambda _message: None,
        inferencer=FakeInferencer(),
    )

    world = emitted[0][0]["world"]
    assert world[11][3] == pytest.approx(coco_scores[5])
    assert world[13][3] == pytest.approx(coco_scores[7])
    assert world[15][3] == pytest.approx(coco_scores[9])
    assert world[23][3] == pytest.approx(coco_scores[11])
    assert world[24][3] == pytest.approx(coco_scores[12])




def test_motionbert_video_reports_progress_and_estimated_time():
    joints = np.zeros((17, 3), dtype=np.float32)
    scores = np.ones(17, dtype=np.float32)

    class FakeInferencer:
        def __call__(self, _video_path):
            for _ in range(16):
                yield {
                    "predictions": [[{
                        "keypoints": joints,
                        "keypoint_scores": scores,
                    }]]
                }

    statuses = []
    emitted = []
    progress_updates = []
    timestamps = [index / 30.0 for index in range(16)]
    count = process_motionbert_video(
        "capture.avi",
        timestamps,
        lambda pose, timestamp: emitted.append((pose, timestamp)),
        statuses.append,
        inferencer=FakeInferencer(),
        progress_callback=lambda *values: progress_updates.append(values),
    )

    assert count == 16
    assert len(emitted) == 16
    assert any("15/16" in status for status in statuses)
    assert any("16/16" in status for status in statuses)
    assert any("FPS" in status and "restan aprox." in status for status in statuses)
    assert progress_updates
    assert progress_updates[-1][:2] == (16, 16)
    assert progress_updates[-1][2] > 0
    assert progress_updates[-1][3] >= 0


def test_mmpose_metadata_lookup_accepts_pyinstaller_repo_layout(tmp_path):
    package_dir = tmp_path / "mmpose"
    package_dir.mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "model-index.yml").write_text("Import: []\n", encoding="utf-8")

    assert _mmpose_metadata_directory(package_dir) == tmp_path


def test_mmpose_metadata_lookup_accepts_installed_mim_layout(tmp_path):
    package_dir = tmp_path / "mmpose"
    mim_dir = package_dir / ".mim"
    (mim_dir / "configs").mkdir(parents=True)
    (mim_dir / "model-index.yml").write_text("Import: []\n", encoding="utf-8")

    assert _mmpose_metadata_directory(package_dir) == mim_dir


def test_mmpose_metadata_lookup_rejects_incomplete_layout(tmp_path):
    package_dir = tmp_path / "mmpose"
    package_dir.mkdir()
    (package_dir.parent / "configs").mkdir()
    # MMEngine needs both the configs directory and its model-index.yml.

    assert _mmpose_metadata_directory(package_dir) is None

def test_motionbert_flip_testing_is_disabled_for_cpu_inference():
    from types import SimpleNamespace

    test_cfg = {
        "flip_test": True,
        "shift_coords": True,
    }
    inferencer = SimpleNamespace(
        inferencer=SimpleNamespace(
            model=SimpleNamespace(test_cfg=test_cfg)
        )
    )

    assert _disable_motionbert_flip_test(inferencer)
    assert test_cfg == {
        "flip_test": False,
        "shift_coords": True,
    }


def test_motionbert_flip_testing_handles_models_without_test_config():
    from types import SimpleNamespace

    inferencer = SimpleNamespace(
        inferencer=SimpleNamespace(model=SimpleNamespace())
    )

    assert not _disable_motionbert_flip_test(inferencer)

