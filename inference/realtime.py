from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import yaml

from models.detector import PlateDetector
from models.pipeline import LPRPipeline, PlateResult, PlateVoter
from models.recognizer import PlateRecognizer
from utils.database import VehicleDB
from utils.overlay import TextItem, TextRenderer, draw_panel
from utils.plate_utils import format_plate_display
from utils.plate_saver import PlateSaver

GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
GREY = (180, 180, 180)


class FrameGrabber:
    """Reads the camera on its own thread and keeps only the newest frame.

    An RTSP camera buffers frames the reader does not consume. If detection is
    slower than the stream, the queue grows and the gate ends up recognising a
    truck that left minutes ago. Dropping stale frames keeps the display live.
    """

    def __init__(self, source, width: int, height: int, reconnect_delay: float = 2.0) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.reconnect_delay = reconnect_delay
        self._frame: Optional[np.ndarray] = None
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

        if cap is not None:
            cap.release()

    def start(self) -> "FrameGrabber":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)


class RealtimeLPR:
    def __init__(self, config_path: str | Path) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        device = self.cfg.get("device", "cpu")
        det_cfg = self.cfg["detector"]
        rec_cfg = self.cfg["recognizer"]
        cam_cfg = self.cfg["camera"]
        db_cfg = self.cfg["database"]
        disp_cfg = self.cfg["display"]
        pre_cfg = self.cfg.get("preprocessing", {})
        cap_cfg = self.cfg.get("capture", {})

        self.detector = PlateDetector(
            model_path=det_cfg["model_path"],
            conf_threshold=det_cfg["conf_threshold"],
            iou_threshold=det_cfg["iou_threshold"],
            img_size=det_cfg["img_size"],
            device=device,
            pad_ratio=det_cfg.get("pad_ratio", 0.06),
            max_det=det_cfg.get("max_det", 8),
            half=det_cfg.get("half", True),
        )
        self.recognizer = PlateRecognizer(
            model_path=rec_cfg["model_path"],
            charset=rec_cfg["charset"],
            backbone=rec_cfg.get("backbone", "resnet34"),
            img_height=rec_cfg["img_height"],
            img_width=rec_cfg["img_width"],
            device=device,
            half=rec_cfg.get("half", True),
        )

        self.db = VehicleDB(db_cfg["path"])
        if db_cfg.get("seed_demo", False):
            self.db.seed_demo()

        saver = None
        if cap_cfg.get("enabled", True):
            saver = PlateSaver(
                output_dir=cap_cfg.get("output_dir", "captures"),
                save_raw=cap_cfg.get("save_raw", True),
                save_enhanced=cap_cfg.get("save_enhanced", True),
                min_interval_seconds=cap_cfg.get("min_interval_seconds", 3.0),
            )

        self.pipeline = LPRPipeline(
            detector=self.detector,
            recognizer=self.recognizer,
            db=self.db,
            deskew=pre_cfg.get("deskew_enabled", True),
            # These were previously read from config and then never used: the
            # pipeline always fell back to enhance_plate's hardcoded defaults.
            enhance_params={
                "clip_limit": pre_cfg.get("clahe_clip", 3.0),
                "bilateral_d": pre_cfg.get("bilateral_d", 9),
                "sigma": pre_cfg.get("bilateral_sigma", 75),
                "sharpen": pre_cfg.get("sharpen_enabled", True),
                "auto": pre_cfg.get("auto_enhance", True),
            },
            voter=PlateVoter(
                window_seconds=pre_cfg.get("vote_window_seconds", 2.0),
                min_votes=pre_cfg.get("min_votes", 3),
                min_score=pre_cfg.get("min_vote_score", 1.5),
            ),
        )

        self.grabber = FrameGrabber(cam_cfg["source"], cam_cfg["width"], cam_cfg["height"])
        self.window_name = disp_cfg["window_name"]
        self.fullscreen = disp_cfg.get("fullscreen", True)
        self.base_size = int(disp_cfg.get("font_scale", 1.1) * 26)
        self.thickness = disp_cfg["thickness"]
        self.hold_seconds = float(disp_cfg.get("hold_seconds", 5.0))
        self.renderer = TextRenderer(disp_cfg.get("font_path"))

        self.last_result: Optional[PlateResult] = None
        self.last_confirmed_at = 0.0
        self._fps = 0.0

    def _held_result(self, result: PlateResult) -> Optional[PlateResult]:
        """Keep a confirmed driver on screen briefly after the truck passes.

        Previously a single frame without a detection blanked the panel, so the
        operator saw the driver's details flicker in and out.
        """
        now = time.monotonic()
        if result.confirmed and result.plate:
            self.last_result = result
            self.last_confirmed_at = now
            return result
        if self.last_result and now - self.last_confirmed_at <= self.hold_seconds:
            return self.last_result
        self.last_result = None
        return None

    def _draw_overlay(self, frame: np.ndarray, result: PlateResult, held: Optional[PlateResult]) -> np.ndarray:
        out = frame.copy()
        info = held.info if held else None

        if result.bbox is not None:
            x1, y1, x2, y2 = result.bbox
            if info is None:
                color = YELLOW
            else:
                color = GREEN if info.get("allowed") else RED
            cv2.rectangle(out, (x1, y1), (x2, y2), color, self.thickness)

        items: List[TextItem] = []
        line_h = self.base_size + 10
        y = 24
        plate_text = held.plate if held else result.plate

        if plate_text or info:
            panel_lines = 1 + (7 if info else 0)
            draw_panel(out, (12, 12), (12 + 30 * self.base_size, 24 + panel_lines * line_h))

        if plate_text:
            label = format_plate_display(plate_text)
            status = "" if (held and held.confirmed) else "  (در حال تایید)"
            items.append(
                (f"پلاک: {label}  [{result.ocr_conf:.2f}]{status}", (24, y), self.base_size, YELLOW)
            )
            y += line_h

        if info:
            allowed = bool(info.get("allowed"))
            for label, key in (
                ("راننده", "driver_name"),
                ("کد ملی", "national_id"),
                ("شماره کامیون", "truck_id"),
                ("مدل", "vehicle_model"),
                ("شرکت", "company"),
            ):
                items.append((f"{label}: {info.get(key) or '-'}", (24, y), int(self.base_size * 0.85), WHITE))
                y += line_h
            items.append(
                (
                    f"وضعیت: {'مجاز' if allowed else 'غیرمجاز'}",
                    (24, y),
                    self.base_size,
                    GREEN if allowed else RED,
                )
            )
            y += line_h
            if info.get("note"):
                items.append((f"توضیح: {info['note']}", (24, y), int(self.base_size * 0.8), GREY))
        elif plate_text:
            items.append(("این پلاک در سامانه ثبت نشده است", (24, y), int(self.base_size * 0.85), RED))

        status = "دوربین متصل" if self.grabber.connected else "قطع ارتباط دوربین"
        items.append(
            (
                f"{status}   |   {self._fps:.1f} FPS",
                (24, out.shape[0] - line_h - 12),
                int(self.base_size * 0.7),
                GREY if self.grabber.connected else RED,
            )
        )
        return self.renderer.render(out, items)

    def run(self) -> None:
        self.grabber.start()
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if self.fullscreen:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        last_tick = time.monotonic()
        try:
            while True:
                frame = self.grabber.read()
                if frame is None:
                    # Camera not up yet: keep the window responsive instead of
                    # exiting, which is what the old loop did on the first drop.
                    if cv2.waitKey(50) & 0xFF in (ord("q"), 27):
                        break
                    continue

                result = self.pipeline.process(frame)
                held = self._held_result(result)

                now = time.monotonic()
                dt = now - last_tick
                last_tick = now
                if dt > 0:
                    self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt) if self._fps else 1.0 / dt

                cv2.imshow(self.window_name, self._draw_overlay(frame, result, held))
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.grabber.stop()
            self.db.close()
            cv2.destroyAllWindows()
