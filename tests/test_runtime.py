import sys

import pytest

import main


@pytest.mark.parametrize(
    "missing_streams",
    [
        ("stderr",),
        ("stdout",),
        ("stdout", "stderr"),
    ],
)
def test_windowed_stdio_is_restored_and_logged(tmp_path, monkeypatch, missing_streams):
    monkeypatch.setattr(main, "_stdio_log_handle", None)
    monkeypatch.setattr(main, "_stdio_log_path", None)
    for stream_name in missing_streams:
        monkeypatch.setattr(sys, stream_name, None)

    log_path = main._configure_stdio(tmp_path)

    assert log_path == tmp_path / "runtime.log"
    assert log_path.is_file()
    for stream_name in missing_streams:
        stream = getattr(sys, stream_name)
        assert stream is not None
        stream.write(f"{stream_name} recovered\\n")

    handle = main._stdio_log_handle
    handle.flush()
    assert "recovered" in log_path.read_text(encoding="utf-8")
    handle.close()


def test_stdio_configuration_does_not_replace_existing_streams(monkeypatch):
    import io

    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    assert main._configure_stdio() is None
    assert sys.stdout is stdout
    assert sys.stderr is stderr
