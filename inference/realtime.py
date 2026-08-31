from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from models.detector import PlateDetector
from models.recognizer import PlateRecognizer
from models.pipeline import LPRPipeline
from utils.database import VehicleDB
from utils.plate_utils import format_plate_display


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

        self.detector = PlateDetector(
            model_path=det_cfg["model_path"],
            conf_threshold=det_cfg["conf_threshold"],
            iou_threshold=det_cfg["iou_threshold"],
            img_size=det_cfg["img_size"],
            device=device,
        )
        self.recognizer = PlateRecognizer(
            model_path=rec_cfg["model_path"],
            charset=rec_cfg["charset"],
            backbone=rec_cfg.get("backbone", "resnet34"),
            img_height=rec_cfg["img_height"],
            img_width=rec_cfg["img_width"],
            device=device,
            pretrained=rec_cfg.get("pretrained", True),
        )
        self.db = VehicleDB(db_cfg["path"])
        self.db.seed_demo()

        self.pipeline = LPRPipeline(
            detector=self.detector,
            recognizer=self.recognizer,
            db=self.db,
            deskew=self.cfg["preprocessing"].get("deskew_enabled", True),
        )

        self.source = cam_cfg["source"]
        self.width = cam_cfg["width"]
        self.height = cam_cfg["height"]
        self.window_name = disp_cfg["window_name"]
        self.font_scale = disp_cfg["font_scale"]
        self.thickness = disp_cfg["thickness"]

        self.last_plate: Optional[str] = None
        self.last_info: Optional[dict] = None
        self.stable_count = 0
        self.stable_threshold = 3

    def _draw_overlay(
        self,
        frame: np.ndarray,
        plate: Optional[str],
        info: Optional[dict],
        bbox: Optional[tuple],
        conf: float,
    ) -> np.ndarray:
        out = frame.copy()
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0) if info and info.get("allowed") else (0, 0, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        y0 = 40
        if plate:
            display = format_plate_display(plate)
            cv2.putText(
                out,
                f"Plate: {display}  ({conf:.2f})",
                (20, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (0, 255, 255),
                self.thickness,
                cv2.LINE_AA,
            )
            y0 += 40

        if info:
            lines = [
                f"Driver : {info.get('driver_name', '-')}",
                f"National ID : {info.get('national_id', '-')}",
                f"Truck ID : {info.get('truck_id', '-')}",
                f"Model : {info.get('vehicle_model', '-')}",
                f"Company : {info.get('company', '-')}",
                f"Status : {'ALLOWED' if info.get('allowed') else 'DENIED'}",
            ]
            if info.get("note"):
                lines.append(f"Note : {info.get('note')}")
            for line in lines:
                cv2.putText(
                    out,
                    line,
                    (20, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale * 0.75,
                    (255, 255, 255),
                    self.thickness,
                    cv2.LINE_AA,
                )
                y0 += 32
        return out

    def run(self) -> None:
        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {self.source}")

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            plate, info, bbox, conf = self.pipeline.process(frame)

            if plate and plate == self.last_plate:
                self.stable_count += 1
            else:
                self.stable_count = 1
                self.last_plate = plate
                self.last_info = info

            show_plate = (
                self.last_plate if self.stable_count >= self.stable_threshold else None
            )
            show_info = (
                self.last_info if self.stable_count >= self.stable_threshold else None
            )

            display = self._draw_overlay(frame, show_plate, show_info, bbox, conf)
            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
