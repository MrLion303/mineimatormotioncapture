import queue

from PySide6.QtGui import QImage

from mocap_app.camera import CameraWorker


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
