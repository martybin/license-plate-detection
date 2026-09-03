"""Live check with the laptop webcam: hold a plate up and see what the model reads.

This is a manual smoke test, not part of the automated suite -- it needs a camera,
a screen and a human holding a plate. Run it to sanity-check trained weights
before taking them to the mine gate.

    python -m tools.live_test                  # webcam, full detector + OCR
    python -m tools.live_test --ocr-only       # skip the detector, read the box
    python -m tools.live_test --image p.jpg    # a single still image
    python -m tools.live_test --source 1       # a second camera

Keys:  q / Esc quit      s save a snapshot      space pause      r reset votes
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.pipeline import LPRPipeline, PlateVoter  # noqa: E402
from utils.image_processing import enhance_plate, estimate_quality  # noqa: E402
from utils.overlay import TextItem, TextRenderer, draw_panel  # noqa: E402
from utils.plate_utils import format_plate_display, is_valid_iran_plate, repair_plate  # noqa: E402

GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
GREY = (170, 170, 170)


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def require_weights(cfg: dict, need_detector: bool) -> None:
    missing = []
    if need_detector and not Path(cfg["detector"]["model_path"]).exists():
        missing.append(f"  detector   : {cfg['detector']['model_path']}")
    if not Path(cfg["recognizer"]["model_path"]).exists():
        missing.append(f"  recognizer : {cfg['recognizer']['model_path']}")
    if missing:
        raise SystemExit(
            "Trained weights are missing:\n"
            + "\n".join(missing)
            + "\n\nTrain them first:\n"
            "  python -m training.prepare_dataset --root /mnt/g/Bistun-kavir\n"
            "  python -m training.train_detector\n"
            "  python -m training.train_recognizer\n"
            + ("\nOr pass --ocr-only to test the recognizer without a detector.\n" if need_detector else "")
        )


def centre_box(frame: np.ndarray, width_frac: float = 0.7, aspect: float = 4.5):
    """The region --ocr-only reads: a plate-shaped box in the middle of the frame."""
    h, w = frame.shape[:2]
    bw = int(w * width_frac)
    bh = max(1, int(bw / aspect))
    if bh > h * 0.8:
        bh = int(h * 0.8)
        bw = int(bh * aspect)
    x1 = (w - bw) // 2
    y1 = (h - bh) // 2
    return x1, y1, x1 + bw, y1 + bh


class LiveTester:
    def __init__(self, args) -> None:
        self.args = args
        self.cfg = load_config(Path(args.config))
        require_weights(self.cfg, need_detector=not args.ocr_only)

        device = self.cfg.get("device", "cpu")
        rec_cfg = self.cfg["recognizer"]
        pre_cfg = self.cfg.get("preprocessing", {})

        from models.recognizer import PlateRecognizer

        print("Loading recognizer ...")
        self.recognizer = PlateRecognizer(
            model_path=rec_cfg["model_path"],
            charset=rec_cfg["charset"],
            backbone=rec_cfg.get("backbone", "resnet18"),
            img_height=rec_cfg["img_height"],
            img_width=rec_cfg["img_width"],
            device=device,
            half=rec_cfg.get("half", True),
        )

        self.enhance_params = {
            "clip_limit": pre_cfg.get("clahe_clip", 3.0),
            "bilateral_d": pre_cfg.get("bilateral_d", 9),
            "sigma": pre_cfg.get("bilateral_sigma", 75),
            "sharpen": pre_cfg.get("sharpen_enabled", True),
            "auto": pre_cfg.get("auto_enhance", True),
        }

        self.pipeline: Optional[LPRPipeline] = None
        if not args.ocr_only:
            from models.detector import PlateDetector
            from utils.database import VehicleDB

            det_cfg = self.cfg["detector"]
            print("Loading detector ...")
            detector = PlateDetector(
                model_path=det_cfg["model_path"],
                conf_threshold=args.conf or det_cfg["conf_threshold"],
                iou_threshold=det_cfg["iou_threshold"],
                img_size=det_cfg["img_size"],
                device=device,
                pad_ratio=det_cfg.get("pad_ratio", 0.06),
                half=det_cfg.get("half", True),
            )
            self.db = VehicleDB(self.cfg["database"]["path"])
            self.pipeline = LPRPipeline(
                detector=detector,
                recognizer=self.recognizer,
                db=self.db,
                deskew=pre_cfg.get("deskew_enabled", True),
                enhance_params=self.enhance_params,
                voter=PlateVoter(
                    window_seconds=pre_cfg.get("vote_window_seconds", 2.0),
                    min_votes=pre_cfg.get("min_votes", 3),
                    min_score=pre_cfg.get("min_vote_score", 1.5),
                ),
            )
        else:
            self.db = None
            self.voter = PlateVoter(min_votes=args.min_votes, min_score=1.0)

        self.renderer = TextRenderer(self.cfg.get("display", {}).get("font_path"))
        self.snapshot_dir = Path(args.snapshot_dir)
        self.fps = 0.0

    # ------------------------------------------------------------------ modes

    def _read_centre(self, frame: np.ndarray):
        """OCR the middle box directly, with no detector in the way."""
        x1, y1, x2, y2 = centre_box(frame)
        crop = frame[y1:y2, x1:x2]
        variants = [crop, enhance_plate(crop, **self.enhance_params)]

        from models.recognizer import Recognition

        readings = [
            Recognition(repair_plate(r.text), r.confidence)
            for r in self.recognizer.recognize_batch(variants)
            if r is not None
        ]
        if not readings:
            return None, (x1, y1, x2, y2), crop

        valid = [r for r in readings if is_valid_iran_plate(r.text)]
        best = max(valid or readings, key=lambda r: r.confidence)
        if is_valid_iran_plate(best.text):
            self.voter.add(best.text, best.confidence)
        return best, (x1, y1, x2, y2), crop

    # ----------------------------------------------------------------- render

    def _draw(self, frame, reading, bbox, confirmed, info, quality):
        out = frame.copy()
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            valid = reading is not None and is_valid_iran_plate(reading.text)
            colour = GREEN if confirmed else (YELLOW if valid else RED)
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

        items: List[TextItem] = []
        size = 26
        line_h = size + 10
        y = 20
        lines = 4 + (2 if info else 0)
        draw_panel(out, (12, 12), (12 + 26 * size, 20 + lines * line_h))

        if reading is not None and reading.text:
            valid = is_valid_iran_plate(reading.text)
            label = format_plate_display(reading.text) if valid else reading.text
            state = "تایید شد" if confirmed else ("معتبر" if valid else "نامعتبر")
            items.append(
                (f"پلاک: {label}", (24, y), size, GREEN if confirmed else (YELLOW if valid else RED))
            )
            y += line_h
            items.append((f"اطمینان: {reading.confidence:.2f}   وضعیت: {state}", (24, y), int(size * 0.8), WHITE))
        else:
            items.append(("پلاکی خوانده نشد", (24, y), size, GREY))
            y += line_h
            items.append(("پلاک را جلوی دوربین بگیرید", (24, y), int(size * 0.8), GREY))
        y += line_h

        if quality is not None:
            items.append(
                (
                    f"روشنایی {quality['brightness']:.0f}  وضوح {quality['blur']:.0f}  "
                    f"بازتاب {quality['glare']:.1%}",
                    (24, y),
                    int(size * 0.72),
                    GREY,
                )
            )
            y += line_h

        if info:
            items.append((f"راننده: {info.get('driver_name') or '-'}", (24, y), int(size * 0.85), WHITE))
            y += line_h
            allowed = bool(info.get("allowed"))
            items.append(
                (f"وضعیت: {'مجاز' if allowed else 'غیرمجاز'}", (24, y), size, GREEN if allowed else RED)
            )
            y += line_h

        mode = "فقط OCR" if self.args.ocr_only else "دتکتور + OCR"
        items.append(
            (f"{mode}  |  {self.fps:.1f} FPS  |  q خروج, s ذخیره, space توقف", (24, y), int(size * 0.7), GREY)
        )
        return self.renderer.render(out, items)

    def _snapshot(self, frame, reading) -> None:
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        plate = reading.text if reading is not None and reading.text else "unread"
        safe = "".join(c if c.isalnum() else "_" for c in plate)
        path = self.snapshot_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{safe}.jpg"
        cv2.imwrite(str(path), frame)
        print(f"saved {path}")

    # -------------------------------------------------------------------- run

    def run_image(self, path: Path) -> None:
        frame = cv2.imread(str(path))
        if frame is None:
            raise SystemExit(f"Cannot read image: {path}")
        reading, bbox, info, confirmed, quality = self._process(frame)
        self._report(reading, confirmed, quality)
        cv2.imshow("LPR live test", self._draw(frame, reading, bbox, confirmed, info, quality))
        print("\nAny key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _process(self, frame):
        if self.args.ocr_only:
            reading, bbox, crop = self._read_centre(frame)
            confirmed = reading is not None and self.voter.confirmed() == reading.text
            quality = estimate_quality(crop) if crop is not None and crop.size else None
            return reading, bbox, None, confirmed, quality

        result = self.pipeline.process(frame)
        reading = None
        if result.plate:
            from models.recognizer import Recognition

            reading = Recognition(result.plate, result.ocr_conf)
        quality = estimate_quality(result.crop) if result.crop is not None and result.crop.size else None
        return reading, result.bbox, result.info, result.confirmed, quality

    def _report(self, reading, confirmed, quality) -> None:
        if reading is None or not reading.text:
            print("no plate read")
            return
        valid = is_valid_iran_plate(reading.text)
        print(f"plate      : {format_plate_display(reading.text) if valid else reading.text}")
        print(f"confidence : {reading.confidence:.3f}")
        print(f"valid      : {valid}   confirmed: {confirmed}")
        if quality:
            print(
                f"quality    : brightness {quality['brightness']:.0f}, "
                f"blur {quality['blur']:.0f}, glare {quality['glare']:.1%}"
            )

    def run_camera(self) -> None:
        source = int(self.args.source) if str(self.args.source).isdigit() else self.args.source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise SystemExit(
                f"Cannot open camera source: {source}\n"
                "On Windows try --source 0 or 1; on WSL the laptop camera is usually "
                "not passed through, so run this from Windows or use --image."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg["camera"]["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg["camera"]["height"])

        window = "LPR live test"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        print("\nHold a plate in front of the camera.  q quit | s save | space pause | r reset\n")

        paused = False
        frame = None
        last_tick = time.monotonic()
        last_print = ""

        try:
            while True:
                if not paused:
                    ok, grabbed = cap.read()
                    if not ok:
                        print("camera read failed")
                        break
                    frame = grabbed

                reading, bbox, info, confirmed, quality = self._process(frame)

                now = time.monotonic()
                dt = now - last_tick
                last_tick = now
                if dt > 0:
                    self.fps = 0.9 * self.fps + 0.1 / dt if self.fps else 1.0 / dt

                if reading is not None and reading.text and reading.text != last_print:
                    self._report(reading, confirmed, quality)
                    print("-" * 40)
                    last_print = reading.text

                cv2.imshow(window, self._draw(frame, reading, bbox, confirmed, info, quality))

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    self._snapshot(frame, reading)
                if key == ord(" "):
                    paused = not paused
                if key == ord("r"):
                    (self.pipeline.voter if self.pipeline else self.voter).reset()
                    last_print = ""
                    print("votes reset")
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            if self.db is not None:
                self.db.close()
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--source", default="0", help="camera index or video path")
    parser.add_argument("--image", default=None, help="run once on a still image instead")
    parser.add_argument(
        "--ocr-only",
        action="store_true",
        help="skip the detector and read the centre box; useful before the detector is trained",
    )
    parser.add_argument("--conf", type=float, default=None, help="override detector confidence threshold")
    parser.add_argument("--min-votes", type=int, default=3, help="votes to confirm in --ocr-only mode")
    parser.add_argument("--snapshot-dir", default="captures/live_test")
    args = parser.parse_args()

    tester = LiveTester(args)
    if args.image:
        tester.run_image(Path(args.image))
    else:
        tester.run_camera()


if __name__ == "__main__":
    main()
