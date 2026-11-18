import pytest

from mocap_app.video import build_video_timeline, pose_timestamp_is_valid


def test_video_timeline_preserves_the_source_playback_speed():
    timestamps, duration = build_video_timeline(300, 30.0)

    assert len(timestamps) == 300
    assert timestamps[0] == 0.0
    assert timestamps[-1] == pytest.approx(299 / 30.0)
    assert duration == pytest.approx(10.0)


def test_video_timeline_supports_non_integer_source_fps():
    timestamps, duration = build_video_timeline(240, 23.976)

    assert len(timestamps) == 240
    assert timestamps[120] == pytest.approx(120 / 23.976)
    assert duration == pytest.approx(240 / 23.976)


@pytest.mark.parametrize(
    ("frame_count", "source_fps", "message"),
    [
        (0, 30.0, "no contiene fotogramas"),
        (120, 0.0, "frecuencia"),
        (120, -24.0, "frecuencia"),
    ],
)
def test_video_timeline_rejects_invalid_metadata(frame_count, source_fps, message):
    with pytest.raises(ValueError, match=message):
        build_video_timeline(frame_count, source_fps)



def test_imported_video_accepts_pose_timestamps_before_duration_is_known():
    assert pose_timestamp_is_valid(0.0, 0.0, 0.0, "video")
    assert pose_timestamp_is_valid(4.5, 0.0, 0.0, "video")


@pytest.mark.parametrize("timestamp", [-0.01, float("nan"), float("inf")])
def test_imported_video_rejects_invalid_pose_timestamps(timestamp):
    assert not pose_timestamp_is_valid(timestamp, 0.0, 0.0, "video")


def test_camera_pose_timestamps_remain_bounded_by_capture_duration():
    assert pose_timestamp_is_valid(9.9, 100.0, 10.0, "camera")
    assert not pose_timestamp_is_valid(10.1, 100.0, 10.0, "camera")
