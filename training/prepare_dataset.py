# training/prepare_dataset.py
"""Build both training datasets from Pascal-VOC style annotations.

The project needs two different datasets from two different image sources:

  * OCR set        - tight plate crops, renamed to the plate text.
                     Source: the IR-LPR "plate" subset (27,745 images).
  * Detection set  - full scenes of cars, labelled with the plate's box.
                     Source: the IR-LPR "car" subset (20,967 images).

Passing plate crops as the detection source produces a detector that only works
on images that are already cropped to a plate, so this script warns loudly when
that happens rather than silently building a useless dataset.
"""
from __future__ import annotations

import argparse
import shutil
import struct
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

from utils.plate_utils import normalize_iran_plate

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Names an annotator may have used for the plate as a whole, as opposed to a
# single character. Compared case-insensitively with separators stripped, so
# 'کل ناحیه پلاک' is matched here without its spaces.
PLATE_OBJECT_NAMES = {
    "plate",
    "licenseplate",
    "license",
    "lp",
    "numberplate",
    "پلاک",
    "کلناحیهپلاک",  # 'کل ناحیه پلاک' - the IR-LPR car subset's whole-plate box
    "ناحیهپلاک",
}

BBoxF = Tuple[float, float, float, float]


class CharBox(NamedTuple):
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    name: str

    @property
    def box(self) -> BBoxF:
        return (self.xmin, self.ymin, self.xmax, self.ymax)


def clean_plate_text(text: str) -> str:
    """Fold to canonical Persian digits/letters.

    Shared with inference via utils.plate_utils so training labels and runtime
    normalisation can never drift apart. It also maps whole-word annotations
    ('معلولین و جانبازان') onto their single plate letter.
    """
    return normalize_iran_plate(text)


# --------------------------------------------------------------------------- #
# Annotation parsing
# --------------------------------------------------------------------------- #

def _find_float(node: ET.Element, tag: str) -> Optional[float]:
    child = node.find(tag)
    if child is None or child.text is None:
        return None
    try:
        return float(child.text)
    except ValueError:
        return None


def image_size(path: Path) -> Optional[Tuple[int, int]]:
    """Read (width, height) from the file header without decoding the pixels.

    The IR-LPR annotations carry no <size> element, so the dimensions have to
    come from the image itself. Parsing the JPEG header directly avoids pulling
    in a decode of all 27,745 files.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return None

    if data[:2] == b"\xff\xd8":  # JPEG
        i = 2
        while i < len(data) - 9:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB):
                height, width = struct.unpack(">HH", data[i + 5 : i + 9])
                return width, height
            if marker == 0xD8 or marker == 0x01 or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            i += 2 + struct.unpack(">H", data[i + 2 : i + 4])[0]
        return None

    if data[:8] == b"\x89PNG\r\n\x1a\n":
        width, height = struct.unpack(">II", data[16:24])
        return int(width), int(height)

    try:  # last resort for BMP and anything exotic
        from PIL import Image

        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def parse_xml(xml_path: Path) -> Tuple[List[CharBox], Optional[Tuple[int, int]]]:
    """Read a Pascal-VOC file into its objects plus the declared image size."""
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, OSError):
        return [], None

    size: Optional[Tuple[int, int]] = None
    size_node = root.find("size")
    if size_node is not None:
        width = _find_float(size_node, "width")
        height = _find_float(size_node, "height")
        if width and height:
            size = (int(width), int(height))

    boxes: List[CharBox] = []
    for obj in root.findall(".//object"):
        name_elem = obj.find("name")
        bndbox = obj.find("bndbox")
        if name_elem is None or bndbox is None or name_elem.text is None:
            continue
        coords = [_find_float(bndbox, tag) for tag in ("xmin", "ymin", "xmax", "ymax")]
        if any(c is None for c in coords):
            continue
        boxes.append(CharBox(*coords, name_elem.text.strip()))  # type: ignore[arg-type]
    return boxes, size


def find_xml(img_path: Path) -> Optional[Path]:
    for suffix in (".xml", ".XML"):
        candidate = img_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def _is_plate_object(name: str) -> bool:
    return "".join(ch for ch in name.lower() if ch.isalnum()) in PLATE_OBJECT_NAMES


def extract_plate_from_xml(xml_path: Path) -> Optional[str]:
    """Read the plate text by ordering the character objects left to right."""
    boxes, _ = parse_xml(xml_path)
    chars = [b for b in boxes if not _is_plate_object(b.name)]
    if not chars:
        return None
    plate = clean_plate_text("".join(b.name for b in sorted(chars, key=lambda b: b.xmin)))
    return plate if len(plate) >= 5 else None


def union_box(boxes: List[CharBox], margin: float = 0.08) -> BBoxF:
    """Union of character boxes, padded outward to approximate the plate border."""
    xmin = min(b.xmin for b in boxes)
    ymin = min(b.ymin for b in boxes)
    xmax = max(b.xmax for b in boxes)
    ymax = max(b.ymax for b in boxes)
    pad_x = (xmax - xmin) * margin
    pad_y = (ymax - ymin) * margin * 2  # glyphs stop short of the plate's edges
    return xmin - pad_x, ymin - pad_y, xmax + pad_x, ymax + pad_y


def plate_regions(xml_path: Path) -> Tuple[List[BBoxF], Optional[Tuple[int, int]], str]:
    """Return every plate box in the image, plus how it was derived.

    An explicit whole-plate object is always preferred: a scene with two cars has
    two plates, and a union of every character across both would span the gap
    between them.
    """
    boxes, size = parse_xml(xml_path)
    if not boxes:
        return [], size, "none"

    explicit = [b.box for b in boxes if _is_plate_object(b.name)]
    if explicit:
        return explicit, size, "explicit"
    return [union_box(boxes)], size, "union"


# --------------------------------------------------------------------------- #
# YOLO detection dataset
# --------------------------------------------------------------------------- #

def write_yolo_label(label_path: Path, regions: List[BBoxF], size: Tuple[int, int]) -> None:
    width, height = size
    lines = []
    for xmin, ymin, xmax, ymax in regions:
        # Clip in pixel space, before normalising. Clamping cx/cy/bw/bh
        # independently would keep a box centred at 0.47 while widening it to the
        # full image, which describes a different rectangle than the annotation.
        xmin = min(max(xmin, 0.0), width)
        ymin = min(max(ymin, 0.0), height)
        xmax = min(max(xmax, 0.0), width)
        ymax = min(max(ymax, 0.0), height)
        if xmax - xmin <= 1 or ymax - ymin <= 1:
            continue
        lines.append(
            f"0 {((xmin + xmax) / 2) / width:.6f} {((ymin + ymax) / 2) / height:.6f} "
            f"{(xmax - xmin) / width:.6f} {(ymax - ymin) / height:.6f}"
        )
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def prepare_detection_dataset(
    train_src: Path,
    val_src: Path,
    test_src: Optional[Path] = None,
    output_root: Path = Path("data/plate_dataset"),
) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)

    dirs = {
        "img_train": output_root / "images" / "train",
        "img_val": output_root / "images" / "val",
        "lbl_train": output_root / "labels" / "train",
        "lbl_val": output_root / "labels" / "val",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, int] = defaultdict(int)
    coverage: List[float] = []

    def process(src: Path, img_dst: Path, lbl_dst: Path) -> int:
        count = 0
        for img_path in sorted(src.glob("*.*")):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue

            xml_path = find_xml(img_path)
            if xml_path is None:
                stats["no_xml"] += 1
                continue

            regions, size, source = plate_regions(xml_path)
            if not regions:
                stats["no_boxes"] += 1
                continue
            if size is None:
                size = image_size(img_path)
            if size is None:
                stats["no_size"] += 1
                continue

            # Names repeat across the train/val/test folders; keep them unique so
            # one split cannot silently overwrite another.
            dest_name = f"{img_path.stem}{img_path.suffix.lower()}"
            if (img_dst / dest_name).exists():
                dest_name = f"{img_path.stem}_{count}{img_path.suffix.lower()}"

            shutil.copy2(img_path, img_dst / dest_name)
            write_yolo_label(lbl_dst / (Path(dest_name).stem + ".txt"), regions, size)

            stats[source] += 1
            area = sum(
                max(0.0, min(x2, size[0]) - max(x1, 0.0)) * max(0.0, min(y2, size[1]) - max(y1, 0.0))
                for x1, y1, x2, y2 in regions
            )
            coverage.append(area / (size[0] * size[1]))
            count += 1
        return count

    n_train = process(train_src, dirs["img_train"], dirs["lbl_train"])
    n_val = process(val_src, dirs["img_val"], dirs["lbl_val"])
    if test_src is not None:
        n_val += process(test_src, dirs["img_val"], dirs["lbl_val"])

    (output_root / "data.yaml").write_text(
        f"path: {output_root.resolve().as_posix()}\n"
        f"train: images/train\nval: images/val\nnames:\n  0: plate\n",
        encoding="utf-8",
    )

    print(f"YOLO detection dataset -> {output_root}")
    print(f"  Train: {n_train} images | Val: {n_val} images")
    print(f"  Boxes from explicit plate objects: {stats['explicit']}")
    print(f"  Boxes from character union       : {stats['union']}")
    for key, label in (("no_xml", "missing XML"), ("no_boxes", "no objects"), ("no_size", "unreadable size")):
        if stats[key]:
            print(f"  Skipped ({label}): {stats[key]}")

    if coverage:
        median = sorted(coverage)[len(coverage) // 2]
        print(f"  Median plate coverage of image area: {median:.1%}")
        if median > 0.35:
            print(
                "\n  *** WARNING ***\n"
                "  The plate fills most of each image, so this source is a set of plate\n"
                "  CROPS, not full scenes. A detector trained on it learns 'the image is\n"
                "  a plate' and will not find plates in a 1280x720 camera frame.\n"
                "  Use the IR-LPR *car* subset as the detection source instead.\n"
            )


# --------------------------------------------------------------------------- #
# OCR dataset
# --------------------------------------------------------------------------- #

def rename_images_using_xml(src_folder: Path, output_folder: Path) -> Tuple[int, int]:
    output_folder.mkdir(parents=True, exist_ok=True)
    name_counter: Dict[str, int] = defaultdict(int)
    success = failed = 0

    for img_path in sorted(src_folder.glob("*.*")):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        xml_path = find_xml(img_path)
        if xml_path is None:
            failed += 1
            continue
        plate = extract_plate_from_xml(xml_path)
        if not plate:
            failed += 1
            continue

        name_counter[plate] += 1
        index = name_counter[plate]
        suffix = img_path.suffix.lower()
        # train_recognizer strips this `_N` back off, so duplicates stay usable.
        new_name = f"{plate}{suffix}" if index == 1 else f"{plate}_{index}{suffix}"
        shutil.copy2(img_path, output_folder / new_name)
        success += 1

    return success, failed


def prepare_ocr_dataset(
    train_src: Path,
    val_src: Path,
    test_src: Optional[Path] = None,
    output_root: Path = Path("data/ocr_dataset"),
) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    total_success = total_failed = 0
    for name, src in (("TRAIN", train_src), ("VALIDATION", val_src), ("TEST", test_src)):
        if src is None:
            continue
        print(f"Processing {name} ...")
        ok, failed = rename_images_using_xml(src, output_root)
        total_success += ok
        total_failed += failed
        print(f"  OK: {ok} | Failed: {failed}")

    print(f"\nOCR dataset -> {output_root}")
    print(f"Total success: {total_success} | Total failed: {total_failed}")


# --------------------------------------------------------------------------- #

def _existing(path: Optional[Path], label: str, required: bool) -> Optional[Path]:
    if path is None:
        return None
    if path.exists():
        return path
    if required:
        raise FileNotFoundError(f"{label} folder not found: {path}")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the YOLO detection and CRNN OCR datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/mnt/g/Bistun-kavir"),
        help="dataset root holding train/validation/test and car_train/car_validation/car_test",
    )

    plates = parser.add_argument_group("plate crops (OCR source)")
    plates.add_argument("--plate-train", type=Path, default=None)
    plates.add_argument("--plate-val", type=Path, default=None)
    plates.add_argument("--plate-test", type=Path, default=None)

    cars = parser.add_argument_group("full scenes (detection source)")
    cars.add_argument("--car-train", type=Path, default=None)
    cars.add_argument("--car-val", type=Path, default=None)
    cars.add_argument("--car-test", type=Path, default=None)

    parser.add_argument("--skip-ocr", action="store_true")
    parser.add_argument("--skip-detection", action="store_true")
    args = parser.parse_args()

    # Explicit paths win; otherwise fall back to the standard layout under --root.
    root = args.root
    plate_train = _existing(args.plate_train or root / "train", "plate train", True)
    plate_val = _existing(args.plate_val or root / "validation", "plate val", True)
    plate_test = _existing(args.plate_test or root / "test", "plate test", False)

    # A path the user typed must exist; only the --root defaults may be absent,
    # otherwise a typo would silently fall back to the plate crops and quietly
    # build the useless detection dataset.
    car_train = _existing(args.car_train or root / "car_train", "car train", args.car_train is not None)
    car_val = _existing(args.car_val or root / "car_validation", "car val", args.car_val is not None)
    car_test = _existing(args.car_test or root / "car_test", "car test", args.car_test is not None)

    if not args.skip_detection:
        print("1) Preparing YOLO detection dataset...")
        if car_train and car_val:
            prepare_detection_dataset(car_train, car_val, car_test)
        else:
            print(
                "  No --car-train/--car-val given; falling back to the plate crops.\n"
                "  This produces a detector that cannot localise plates in a full frame."
            )
            prepare_detection_dataset(plate_train, plate_val, plate_test)

    if not args.skip_ocr:
        print("\n2) Preparing OCR dataset (auto-rename with real plate text)...")
        prepare_ocr_dataset(plate_train, plate_val, plate_test)

    print("\nDONE!")


if __name__ == "__main__":
    main()
