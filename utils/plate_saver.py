from __future__ import annotations

import csv
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import cv2
import numpy as np

from utils.image_processing import imwrite_unicode

CSV_HEADER = [
    "timestamp",
    "plate",
    "det_conf",
    "ocr_conf",
    "full_frame",
    "plate_raw",
    "plate_enhanced",
]

MAX_TRACKED_PLATES = 4096
# Retention sweeps the whole capture tree; run it every N writes, not every one.
RETENTION_EVERY = 50


class _Job(NamedTuple):
    stamp: datetime
    plate: str
    det_conf: float
    ocr_conf: float
    full_frame: Optional[np.ndarray]
    raw_crop: Optional[np.ndarray]
    enhanced_crop: Optional[np.ndarray]
    bbox: Optional[Tuple[int, int, int, int]]


class PlateSaver:
    """Archives a photo of every recognised vehicle so readings can be audited.

    The point of this feature is for an operator to compare what the model read
    against what the vehicle actually was, so the *full frame* is the important
    artefact -- a 140px plate crop on its own does not show which truck it came
    from. The crops are saved alongside it to explain what the OCR actually saw.

    All disk I/O happens on a background thread: encoding three JPEGs takes tens
    of milliseconds, which would otherwise be paid inside the detection loop and
    show up as dropped frames at the gate.
    """

    def __init__(
        self,
        output_dir: str | Path = "captures",
        save_full_frame: bool = True,
        save_raw: bool = True,
        save_enhanced: bool = True,
        annotate: bool = True,
        min_interval_seconds: float = 3.0,
        jpeg_quality: int = 90,
        max_files: int = 0,
        max_age_days: float = 0.0,
        queue_size: int = 32,
        retention_every: int = RETENTION_EVERY,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_full_frame = save_full_frame
        self.save_raw = save_raw
        self.save_enhanced = save_enhanced
        self.annotate = annotate
        # A plate already on screen is not re-saved until it has been absent for
        # this long, so one vehicle pass yields one capture rather than one every
        # few seconds while the truck waits at the barrier.
        self.event_gap = float(min_interval_seconds)
        self.jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
        self.max_files = int(max_files)
        self.max_age_days = float(max_age_days)
        self.retention_every = max(1, int(retention_every))

        self.csv_path = self.output_dir / "captures.csv"
        self.dropped = 0
        self.written = 0

        self._last_seen: Dict[str, float] = {}
        self._queue: "queue.Queue[Optional[_Job]]" = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    # ---------------------------------------------------------------- public

    def save(
        self,
        plate: str,
        raw_crop: Optional[np.ndarray] = None,
        enhanced_crop: Optional[np.ndarray] = None,
        full_frame: Optional[np.ndarray] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        ocr_conf: float = 0.0,
        det_conf: float = 0.0,
    ) -> bool:
        """Queue one capture. Returns True if it was accepted for writing."""
        if not plate:
            return False

        now = time.monotonic()
        if not self._is_new_event(plate, now):
            return False

        job = _Job(
            stamp=datetime.now(),
            plate=plate,
            det_conf=det_conf,
            ocr_conf=ocr_conf,
            # Copy on the caller's thread: the frame buffer is reused for the
            # next camera read, so a reference would be overwritten mid-encode.
            full_frame=full_frame.copy() if (self.save_full_frame and full_frame is not None) else None,
            raw_crop=raw_crop.copy() if (self.save_raw and raw_crop is not None) else None,
            enhanced_crop=(
                enhanced_crop.copy() if (self.save_enhanced and enhanced_crop is not None) else None
            ),
            bbox=bbox,
        )

        try:
            self._queue.put_nowait(job)
        except queue.Full:
            # Never block the gate on a slow disk; losing an audit image is far
            # better than stalling recognition.
            self.dropped += 1
            return False
        return True

    def close(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=5.0)

    def __enter__(self) -> "PlateSaver":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # --------------------------------------------------------------- internal

    def _is_new_event(self, plate: str, now: float) -> bool:
        last = self._last_seen.get(plate)
        self._last_seen[plate] = now
        if last is not None and (now - last) < self.event_gap:
            return False

        # Bound the dedup table: a gate sees thousands of distinct plates over a
        # season and this would otherwise grow for the life of the process.
        if len(self._last_seen) > MAX_TRACKED_PLATES:
            cutoff = now - max(self.event_gap * 4, 300.0)
            self._last_seen = {p: t for p, t in self._last_seen.items() if t >= cutoff}
            if len(self._last_seen) > MAX_TRACKED_PLATES:
                # Everything is still recent, so age alone frees nothing. Drop the
                # oldest outright; without this the "bound" does not bound.
                newest = sorted(self._last_seen.items(), key=lambda kv: kv[1])[-MAX_TRACKED_PLATES:]
                self._last_seen = dict(newest)
            self._last_seen[plate] = now
        return True

    def _writer_loop(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                break
            try:
                self._write(job)
            except Exception:
                # A capture failure must never take the gate down with it.
                pass
            finally:
                self._queue.task_done()

    def _annotated(self, job: _Job) -> np.ndarray:
        frame = job.full_frame
        if not self.annotate or job.bbox is None:
            return frame
        out = frame.copy()
        x1, y1, x2, y2 = job.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Latin only: cv2 cannot draw Persian, and the plate text is already in
        # the filename and the CSV.
        cv2.putText(
            out,
            f"det {job.det_conf:.2f}  ocr {job.ocr_conf:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return out

    def _write(self, job: _Job) -> None:
        # One folder per day keeps a 24/7 gate from producing a single directory
        # with tens of thousands of files in it.
        day_dir = self.output_dir / job.stamp.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)

        safe_plate = "".join(c if (c.isalnum() or c in "-_") else "_" for c in job.plate)
        stem = f"{job.stamp.strftime('%H%M%S_%f')[:-3]}_{safe_plate}"

        written: Dict[str, str] = {}
        for key, image in (
            ("full_frame", self._annotated(job) if job.full_frame is not None else None),
            ("plate_raw", job.raw_crop),
            ("plate_enhanced", job.enhanced_crop),
        ):
            if image is None or image.size == 0:
                continue
            suffix = {"full_frame": "full", "plate_raw": "raw", "plate_enhanced": "enh"}[key]
            path = day_dir / f"{stem}_{suffix}.jpg"
            if imwrite_unicode(path, image, self.jpeg_params):
                # as_posix so the log reads the same on Windows and Linux.
                written[key] = path.relative_to(self.output_dir).as_posix()

        self._append_csv(job, written)
        self.written += 1
        # Retention walks the whole capture tree, so running it on every write
        # would mean a 20000-file stat() sweep per vehicle. Sweep on the first
        # capture -- so a gate that restarts often still prunes -- then amortise.
        if self.max_files or self.max_age_days:
            if self.written == 1 or self.written % self.retention_every == 0:
                self._enforce_retention()

    def _append_csv(self, job: _Job, written: Dict[str, str]) -> None:
        new_file = not self.csv_path.exists()
        with self.csv_path.open("a", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(CSV_HEADER)
            writer.writerow(
                [
                    job.stamp.strftime("%Y-%m-%d %H:%M:%S"),
                    job.plate,
                    f"{job.det_conf:.4f}",
                    f"{job.ocr_conf:.4f}",
                    written.get("full_frame", ""),
                    written.get("plate_raw", ""),
                    written.get("plate_enhanced", ""),
                ]
            )

    def _enforce_retention(self) -> None:
        """Keep the capture folder from filling the disk on a 24/7 gate."""
        # stat() once per file and keep the value; calling it again inside the
        # filters below would triple the syscalls on a 20000-file tree.
        dated = []
        for path in self.output_dir.rglob("*.jpg"):
            try:
                dated.append((path.stat().st_mtime, path))
            except OSError:
                continue
        dated.sort()

        doomed: List[Path] = []
        keep = dated
        if self.max_age_days > 0:
            cutoff = time.time() - self.max_age_days * 86400
            doomed = [p for mtime, p in dated if mtime < cutoff]
            keep = [(m, p) for m, p in dated if m >= cutoff]
        if self.max_files > 0 and len(keep) > self.max_files:
            doomed += [p for _, p in keep[: len(keep) - self.max_files]]

        for path in doomed:
            try:
                path.unlink()
            except OSError:
                pass

        # Drop day folders that retention has emptied.
        for day_dir in self.output_dir.iterdir():
            if day_dir.is_dir() and not any(day_dir.iterdir()):
                try:
                    day_dir.rmdir()
                except OSError:
                    pass
