"""Turn a folder of unlabelled plate photos into training data.

Real site photos arrive with no ground truth -- the filenames are camera
timestamps. This reads them with the trained model and keeps only the readings
it is confident about, so the expensive part (typing plate numbers by hand)
shrinks to reviewing the leftovers.

    # 1. propose labels for a folder of raw captures
    python -m tools.autolabel_plates label --source /mnt/g/Bistun-kavir/plates

    # 2. hand-correct review/needs_review.csv, then pull those in too
    python -m tools.autolabel_plates import review/needs_review.csv

Acceptance is deliberately strict: a reading must be a structurally valid
Iranian plate, clear the confidence floor, and be agreed on by at least two of
the preprocessing variants. A wrong label is worse than no label -- it teaches
the model the wrong glyph -- so the default is to reject and ask a human.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from models.recognizer import PlateRecognizer, Recognition
from utils.image_processing import correct_perspective, enhance_plate, imread_unicode
from utils.plate_utils import is_valid_iran_plate, repair_plate

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
REVIEW_CSV_FIELDS = ["file", "suggested_plate", "confidence", "reason", "corrected_plate"]


def load_config(path: str) -> dict:
    config = Path(path)
    if not config.exists():
        raise SystemExit(
            f"error: config not found at {config.resolve()}\n"
            "Run this from the project root, or pass --config with a full path."
        )
    return yaml.safe_load(config.read_text(encoding="utf-8"))


def build_recognizer(cfg: dict) -> PlateRecognizer:
    rec = cfg["recognizer"]
    weights = Path(rec["model_path"])
    if not weights.exists():
        raise SystemExit(
            f"error: recognizer weights not found at {weights}\n"
            "Train the model first:  python -m training.train_recognizer"
        )
    return PlateRecognizer(
        model_path=weights,
        charset=rec["charset"],
        backbone=rec.get("backbone", "resnet18"),
        img_height=rec["img_height"],
        img_width=rec["img_width"],
        device=cfg.get("device", "cpu"),
        half=rec.get("half", True),
        allowed_letters=rec.get("allowed_letters", ""),
    )


def build_detector(cfg: dict):
    """Optional: re-crop full scenes, or tighten loose crops, before reading."""
    from models.detector import PlateDetector

    det = cfg["detector"]
    weights = Path(det["model_path"])
    if not weights.exists():
        raise SystemExit(
            f"error: detector weights not found at {weights}\n"
            "Train it first:  python -m training.train_detector"
        )
    return PlateDetector(
        model_path=weights,
        conf_threshold=det["conf_threshold"],
        iou_threshold=det["iou_threshold"],
        img_size=det["img_size"],
        device=cfg.get("device", "cpu"),
        pad_ratio=det.get("pad_ratio", 0.06),
        half=det.get("half", True),
    )


def read_with_variants(
    recognizer: PlateRecognizer, crop, enhance_params: dict
) -> Tuple[Optional[Recognition], int]:
    """Read one crop through the same variants the live pipeline uses.

    Returns the best reading and how many variants agreed on its text.
    """
    variants = [crop, enhance_plate(crop, **enhance_params)]
    warped = correct_perspective(crop)
    if warped is not crop and warped.size > 0:
        variants.append(enhance_plate(warped, **enhance_params))

    readings = [r for r in recognizer.recognize_batch(variants) if r is not None]
    if not readings:
        return None, 0

    repaired = [Recognition(repair_plate(r.text), r.confidence) for r in readings]
    valid = [r for r in repaired if is_valid_iran_plate(r.text)]
    best = max(valid or repaired, key=lambda r: r.confidence)
    agreement = sum(1 for r in repaired if r.text == best.text)
    return best, agreement


def next_free_name(out_dir: Path, plate: str, suffix: str, counter: Dict[str, int]) -> Path:
    """Match prepare_dataset's `<plate>_N` convention for repeat sightings."""
    counter[plate] += 1
    index = counter[plate]
    name = f"{plate}{suffix}" if index == 1 else f"{plate}_{index}{suffix}"
    while (out_dir / name).exists():
        counter[plate] += 1
        index = counter[plate]
        name = f"{plate}_{index}{suffix}"
    return out_dir / name


def seed_counter(out_dir: Path) -> Dict[str, int]:
    """Count what is already in the dataset so new files do not collide."""
    counter: Dict[str, int] = defaultdict(int)
    if not out_dir.is_dir():
        return counter
    for path in out_dir.iterdir():
        if path.suffix.lower() not in IMG_EXTS:
            continue
        stem = path.stem
        if "_" in stem:
            head, _, tail = stem.rpartition("_")
            if head and tail.isdigit():
                stem = head
        counter[stem] += 1
    return counter


def cmd_label(args) -> None:
    cfg = load_config(args.config)
    source = Path(args.source)
    if not source.is_dir():
        raise SystemExit(f"error: {source} is not a directory")

    files = sorted(p for p in source.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise SystemExit(f"error: no images under {source}")
    if args.limit:
        files = files[: args.limit]

    recognizer = build_recognizer(cfg)
    detector = build_detector(cfg) if args.detect else None

    pre = cfg.get("preprocessing", {})
    enhance_params = {
        "clip_limit": pre.get("clahe_clip", 3.0),
        "bilateral_d": pre.get("bilateral_d", 9),
        "sigma": pre.get("bilateral_sigma", 75),
        "sharpen": pre.get("sharpen_enabled", True),
        "auto": pre.get("auto_enhance", True),
    }

    out_dir = Path(args.out)
    review_dir = Path(args.review_dir)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        review_dir.mkdir(parents=True, exist_ok=True)

    counter = seed_counter(out_dir)
    stats = Counter()
    review_rows: List[dict] = []
    plates_seen = Counter()

    print(f"Reading {len(files)} images from {source}\n")
    for i, path in enumerate(files, 1):
        if i % 500 == 0:
            print(f"  {i}/{len(files)} ...")

        image = imread_unicode(path)
        if image is None:
            stats["unreadable_file"] += 1
            continue

        crop = image
        if detector is not None:
            best = detector.detect_best(image)
            if best is None:
                stats["no_plate_found"] += 1
                review_rows.append(
                    {"file": str(path), "suggested_plate": "", "confidence": "0.0",
                     "reason": "no_plate_found", "corrected_plate": ""}
                )
                continue
            crop = best.crop

        reading, agreement = read_with_variants(recognizer, crop, enhance_params)
        if reading is None:
            stats["unreadable"] += 1
            reason, plate, conf = "unreadable", "", 0.0
        else:
            plate, conf = reading.text, reading.confidence
            if not is_valid_iran_plate(plate):
                reason = "invalid_format"
            elif conf < args.min_confidence:
                reason = "low_confidence"
            elif agreement < args.min_agreement:
                reason = "variants_disagree"
            else:
                reason = ""

        if reason:
            stats[reason] += 1
            review_rows.append(
                {"file": str(path), "suggested_plate": plate, "confidence": f"{conf:.4f}",
                 "reason": reason, "corrected_plate": ""}
            )
            continue

        stats["accepted"] += 1
        plates_seen[plate] += 1
        if not args.dry_run:
            dest = next_free_name(out_dir, plate, path.suffix.lower(), counter)
            shutil.copy2(path, dest)

    if review_rows and not args.dry_run:
        review_csv = review_dir / "needs_review.csv"
        with review_csv.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=REVIEW_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(review_rows)

    total = len(files)
    accepted = stats["accepted"]
    print(f"\n{'=' * 52}")
    print(f"accepted : {accepted:6d}  ({accepted / total:.1%})")
    for key in ("low_confidence", "variants_disagree", "invalid_format",
                "unreadable", "no_plate_found", "unreadable_file"):
        if stats[key]:
            print(f"{key:<9}: {stats[key]:6d}  ({stats[key] / total:.1%})")
    print(f"{'=' * 52}")
    print(f"distinct plates accepted: {len(plates_seen)}")
    if plates_seen:
        top = plates_seen.most_common(3)
        print(f"most repeated: {', '.join(f'{p} x{n}' for p, n in top)}")

    if args.dry_run:
        print("\n(dry run - nothing written)")
    else:
        print(f"\naccepted images -> {out_dir}")
        if review_rows:
            print(f"needs review    -> {review_dir / 'needs_review.csv'}  ({len(review_rows)} rows)")
            print("\nFill in the 'corrected_plate' column, then:")
            print(f"  python -m tools.autolabel_plates import {review_dir / 'needs_review.csv'}")


def cmd_import(args) -> None:
    """Pull hand-corrected rows from the review CSV into the dataset."""
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise SystemExit(f"error: {csv_path} not found")

    out_dir = Path(args.out)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    counter = seed_counter(out_dir)

    imported = skipped = blank = 0
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = {"file", "corrected_plate"} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"error: CSV needs columns {sorted(missing)}; found {reader.fieldnames}")

        for row in reader:
            corrected = (row.get("corrected_plate") or "").strip()
            if not corrected:
                blank += 1
                continue

            plate = repair_plate(corrected)
            if not is_valid_iran_plate(plate):
                print(f"  skipped (not a valid plate): {corrected!r}", file=sys.stderr)
                skipped += 1
                continue

            src = Path(row["file"])
            if not src.exists():
                print(f"  skipped (file missing): {src}", file=sys.stderr)
                skipped += 1
                continue

            if not args.dry_run:
                shutil.copy2(src, next_free_name(out_dir, plate, src.suffix.lower(), counter))
            imported += 1

    print(f"imported {imported} | skipped {skipped} | left blank {blank}")
    if args.dry_run:
        print("(dry run - nothing written)")


def main() -> None:
    # These are accepted on either side of the subcommand. argparse normally
    # forces parent-level options to come first, which nobody expects:
    # `autolabel_plates label --dry-run` is the natural way to type it.
    # SUPPRESS keeps the subparser copies from clobbering the parent's value
    # when the flag is not repeated.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=argparse.SUPPRESS)
    common.add_argument("--out", default=argparse.SUPPRESS)
    common.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--out", default="data/ocr_dataset", help="where accepted images land")
    parser.add_argument("--dry-run", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    label = sub.add_parser("label", parents=[common],
                           help="read an unlabelled folder and propose labels")
    label.add_argument("--source", required=True)
    label.add_argument("--review-dir", default="review")
    label.add_argument("--min-confidence", type=float, default=0.90)
    label.add_argument("--min-agreement", type=int, default=2,
                       help="how many preprocessing variants must agree (max 3)")
    label.add_argument("--detect", action="store_true",
                       help="run the detector first (for full scenes, or to tighten loose crops)")
    label.add_argument("--limit", type=int, default=0)
    label.set_defaults(func=cmd_label)

    importer = sub.add_parser("import", parents=[common],
                              help="pull hand-corrected rows into the dataset")
    importer.add_argument("csv_path")
    importer.set_defaults(func=cmd_import)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
