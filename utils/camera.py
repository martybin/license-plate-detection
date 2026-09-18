from __future__ import annotations
import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np

class FrameGrabber:
    """Reads the camera on its own thread and keeps only the newest frame.

    An RTSP camera buffers frames the reader does not consume. If detection is
    slower than the stream, the queue grows and the gate ends up recognising a
    truck that left minutes ago. Dropping stale frames keeps the display live.
    """

    def __init__(self, source, width: int, height: int, fps: int = 0,
                 reconnect_delay: float = 2.0) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps = int(fps or 0)
        self.reconnect_delay = reconnect_delay
        self._frame: Optional[np.ndarray] = None
        # Monotonic counter so the consumer can tell a fresh frame from the one
        # it already processed. Without it the loop re-reads the same buffer.
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.connected = False

    def _open(self) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps > 0:
            # Asking the camera for the configured rate stops the grab thread
            # spinning faster than the sensor actually delivers.
            cap.set(cv2.CAP_PROP_FPS, self.fps)
        # Keep the driver-side buffer minimal so we stay close to real time.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _loop(self) -> None:
        cap: Optional[cv2.VideoCapture] = None
        while not self._stop.is_set():
            if cap is None:
                cap = self._open()
                if cap is None:
                    self.connected = False
                    # A dropped link at a mine gate must not kill the process;
                    # keep retrying until the camera comes back.
                    self._stop.wait(self.reconnect_delay)
                    continue
                self.connected = True

            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                cap = None
                self.connected = False
                continue

            with self._lock:
                self._frame = frame
                self._seq += 1

        if cap is not None:
            cap.release()

    def start(self) -> "FrameGrabber":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def read(self) -> Tuple[Optional[np.ndarray], int]:
        """Return the newest frame and its sequence number.

        The caller compares the sequence against the last one it processed. The
        camera delivers ~30 frames a second while the loop can spin far faster,
        so without this the same frame is recognised repeatedly and each pass
        casts another vote -- letting one frame alone satisfy `min_votes`.
        """
        with self._lock:
            if self._frame is None:
                return None, self._seq
            return self._frame.copy(), self._seq

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

