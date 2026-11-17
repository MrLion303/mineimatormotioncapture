import queue

from PySide6.QtGui import QImage

from mocap_app.camera import CameraWorker, _strict_capture_timestamp


def test_inference_queue_keeps_the_most_recent_frame():
    pending = queue.Queue(maxsize=1)
    first_frame = object()
    newest_frame = object()

    CameraWorker._enqueue_latest(pending, (first_frame, 1.0, 1000))
    CameraWorker._enqueue_latest(pending, (newest_frame, 1.1, 1100))

    frame, captured_at, timestamp_ms = pending.get_nowait()
    assert frame is newest_frame
    assert captured_at == 1.1
    assert timestamp_ms == 1100
    assert pending.empty()


def test_preview_frame_counter_publishes_latest_image():
    worker = CameraWorker(0)
    image = QImage(2, 2, QImage.Format.Format_RGB32)

    worker._publish_preview_frame(image)
    first = worker.latest_preview_frame()
    worker._publish_preview_frame(image)
    second = worker.latest_preview_frame()

    assert first is not None
    assert second is not None
    assert first[0] == 1
    assert second[0] == 2
    assert second[1] == image


def test_recording_queue_keeps_recent_chronological_samples():
    pending = queue.Queue(maxsize=2)
    frames = [object(), object(), object()]
    for index, frame in enumerate(frames):
        CameraWorker._enqueue_recording(
            pending,
            (frame, float(index), 1000 + index),
        )

    first = pending.get_nowait()
    second = pending.get_nowait()
    assert first[0] is frames[1]
    assert second[0] is frames[2]


def test_latest_queue_discards_backlog_outside_recording():
    pending = queue.Queue(maxsize=5)
    frames = [object(), object(), object()]
    for index, frame in enumerate(frames):
        CameraWorker._enqueue_latest(
            pending,
            (frame, float(index), 1000 + index),
        )

    assert pending.qsize() == 1
    assert pending.get_nowait()[0] is frames[-1]



def test_recording_timestamps_stay_increasing_when_clock_values_repeat():
    repeated_clock_value = 123.4567
    timestamps = [
        _strict_capture_timestamp(repeated_clock_value, 123456),
        _strict_capture_timestamp(repeated_clock_value, 123457),
        _strict_capture_timestamp(repeated_clock_value, 123458),
    ]

    assert timestamps == [123.4567, 123.457, 123.458]
    assert all(current > previous for previous, current in zip(timestamps, timestamps[1:]))


def test_recording_capture_state_is_read_from_worker():
    worker = CameraWorker(0)

    assert not worker._is_recording_capture()

    worker.begin_recording_capture()
    assert worker._is_recording_capture()

    worker.finish_recording_capture()
    assert not worker._is_recording_capture()


def test_strict_timestamp_also_protects_against_clock_regression():
    first = _strict_capture_timestamp(50.0, 50000)
    second = _strict_capture_timestamp(49.0, 49999, first)
    third = _strict_capture_timestamp(49.0, 49999, second)

    assert first == 50.0
    assert second > first
    assert third > second
