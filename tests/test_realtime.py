"""Camera loop plumbing: frame freshness, reconnection and the display hold."""
from __future__ import annotations

import time

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("torch", reason="inference.realtime imports the model stack")

try:
    from inference.realtime import FrameGrabber, RealtimeLPR
    from models.pipeline import PlateResult
except Exception as exc:  # ultralytics probes the host OS and can fail on import
    pytest.skip(f"inference.realtime unavailable: {exc}", allow_module_level=True)


@pytest.fixture
def video(tmp_path):
    """A short synthetic clip, each frame a different shade."""
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (160, 120))
    assert writer.isOpened(), "no MJPG encoder available"
    for i in range(12):
        writer.write(np.full((120, 160, 3), 10 + i * 15, dtype=np.uint8))
    writer.release()
    return path


def wait_for_frame(grabber, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame, seq = grabber.read()
        if frame is not None:
            return frame, seq
        time.sleep(0.02)
    return None, -1


class TestFrameGrabber:
    def test_reads_a_source(self, video):
        grabber = FrameGrabber(str(video), 160, 120).start()
        try:
            frame, seq = wait_for_frame(grabber)
            assert frame is not None
            assert frame.shape == (120, 160, 3)
            assert seq > 0
        finally:
            grabber.stop()

    def test_sequence_marks_a_frame_as_already_seen(self, video):
        """Two reads with no new frame in between must report the same sequence.

        The loop uses this to decide whether to run recognition. Without it the
        same frame is recognised repeatedly and votes for itself each time, so a
        single frame satisfies `min_votes` and multi-frame fusion does nothing.
        """
        grabber = FrameGrabber(str(video), 160, 120).start()
        try:
            wait_for_frame(grabber)
            grabber._stop.set()  # freeze the producer
            grabber._thread.join(timeout=3.0)

            _, first = grabber.read()
            _, second = grabber.read()
            _, third = grabber.read()
            assert first == second == third
        finally:
            grabber.stop()

    def test_sequence_advances_on_new_frames(self, video):
        grabber = FrameGrabber(str(video), 160, 120).start()
        try:
            _, first = wait_for_frame(grabber)
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                _, latest = grabber.read()
                if latest > first:
                    break
                time.sleep(0.02)
            assert latest > first
        finally:
            grabber.stop()

    def test_read_before_any_frame(self, tmp_path):
        grabber = FrameGrabber(str(tmp_path / "nope.mp4"), 160, 120)
        frame, seq = grabber.read()
        assert frame is None and seq == 0

    def test_bad_source_does_not_raise_and_reports_disconnected(self, tmp_path):
        """A dead camera must not kill the gate process."""
        grabber = FrameGrabber(str(tmp_path / "missing.mp4"), 160, 120, reconnect_delay=0.05).start()
        try:
            time.sleep(0.3)
            assert grabber.connected is False
            assert grabber.read()[0] is None
            assert grabber._thread.is_alive(), "the grabber must keep retrying"
        finally:
            grabber.stop()

    def test_stop_is_idempotent(self, video):
        grabber = FrameGrabber(str(video), 160, 120).start()
        grabber.stop()
        grabber.stop()
        assert not grabber._thread.is_alive()

    def test_read_returns_a_copy(self, video):
        grabber = FrameGrabber(str(video), 160, 120).start()
        try:
            frame, _ = wait_for_frame(grabber)
            frame[:] = 0
            again, _ = grabber.read()
            assert not np.array_equal(again, frame) or again.sum() > 0
        finally:
            grabber.stop()


def bare_app(hold_seconds=5.0):
    """A RealtimeLPR with only the hold-state attributes; no models loaded."""
    app = RealtimeLPR.__new__(RealtimeLPR)
    app.hold_seconds = hold_seconds
    app.last_result = None
    app.last_confirmed_at = 0.0
    return app


def confirmed(plate="12ب34567"):
    return PlateResult(plate, {"driver_name": "x", "allowed": 1}, (1, 2, 3, 4), 0.9, 0.9, True)


UNCONFIRMED = PlateResult(None, None, None, 0.0, 0.0, False)


class TestDisplayHold:
    def test_nothing_held_initially(self):
        assert bare_app()._held_result() is None

    def test_confirmation_is_held(self):
        app = bare_app()
        app._record_result(confirmed())
        assert app._held_result().plate == "12ب34567"

    def test_survives_frames_without_a_detection(self):
        """One missed frame used to blank the panel, making the details flicker."""
        app = bare_app()
        app._record_result(confirmed())
        app._record_result(UNCONFIRMED)
        assert app._held_result().plate == "12ب34567"

    def test_expires_after_hold_seconds(self):
        app = bare_app(hold_seconds=0.15)
        app._record_result(confirmed())
        assert app._held_result() is not None
        time.sleep(0.2)
        assert app._held_result() is None

    def test_redrawing_does_not_extend_the_hold(self):
        """Recording and querying are separate so a repaint cannot freeze the
        panel on screen forever when the camera drops."""
        app = bare_app(hold_seconds=0.2)
        app._record_result(confirmed())
        for _ in range(6):
            app._held_result()
            time.sleep(0.05)
        assert app._held_result() is None

    def test_unconfirmed_result_is_never_latched(self):
        app = bare_app()
        app._record_result(UNCONFIRMED)
        assert app._held_result() is None

    def test_new_confirmation_replaces_the_old(self):
        app = bare_app()
        app._record_result(confirmed("12ب34567"))
        app._record_result(confirmed("34ج67890"))
        assert app._held_result().plate == "34ج67890"
