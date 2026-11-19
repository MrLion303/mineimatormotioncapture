import math

import pytest

from mocap_app.video import build_video_timeline, pose_timestamp_is_valid


def test_video_timeline_preserves_source_frame_rate_and_duration():
    timestamps, duration = build_video_timeline(300, 25.0)

    assert len(timestamps) == 300
    assert timestamps[0] == 0.0
    assert timestamps[1] == pytest.approx(0.04)
    assert timestamps[-1] == pytest.approx(299 / 25.0)
    assert duration == pytest.approx(12.0)


@pytest.mark.parametrize(
    ("frame_count", "fps"),
    [(0, 30.0), (-1, 30.0), (120, 0.0), (120, -30.0), (120, math.inf), (120, math.nan)],
)
def test_video_timeline_rejects_invalid_properties(frame_count, fps):
    with pytest.raises(ValueError):
        build_video_timeline(frame_count, fps)


def test_imported_video_accepts_timestamps_until_processing_finishes():
    assert pose_timestamp_is_valid(14.5, 0.0, 0.0, "video")
    assert pose_timestamp_is_valid(0.0, 0.0, 0.0, "video")
    assert not pose_timestamp_is_valid(-0.01, 0.0, 0.0, "video")


def test_camera_timestamp_stays_inside_the_recorded_interval():
    assert pose_timestamp_is_valid(2.0, 1.0, 1.0, "camera")
    assert not pose_timestamp_is_valid(2.01, 1.0, 1.0, "camera")
    assert not pose_timestamp_is_valid(0.99, 1.0, 1.0, "camera")
