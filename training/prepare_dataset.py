# training/prepare_dataset.py
from __future__ import annotations

import re
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from PIL import Image

def clean_plate_text(text: str) -> str:
    text = text.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789"))
    text = re.sub(r"[^0-9ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی]", "", text)
    return text

def extract_plate_from_xml(xml_path: Path) -> str | None:
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        objects = []
        for obj in root.findall(".//object"):
            name_elem = obj.find("name")
            bndbox = obj.find("bndbox")
            if name_elem is None or bndbox is None:
                continue
            name = name_elem.text
            if name is None:
                continue
            try:
                xmin = float(bndbox.find("xmin").text)
            except Exception:
                continue
            objects.append((xmin, name.strip()))

        if not objects:
            return None

        objects.sort(key=lambda x: x[0])
        plate = "".join([name for _, name in objects])
        plate = clean_plate_text(plate)
        if len(plate) >= 5:
            return plate
        return None
    except Exception:
        return None

def rename_images_using_xml(src_folder: Path, output_folder: Path) -> tuple[int, int]:
    output_folder.mkdir(parents=True, exist_ok=True)
    name_counter = defaultdict(int)
    success = 0
    failed = 0

    for img_path in src_folder.glob("*.*"):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue

        xml_path = img_path.with_suffix(".xml")
        if not xml_path.exists():
            xml_path = img_path.parent / (img_path.stem + ".XML")
            if not xml_path.exists():
                failed += 1
                continue

        plate = extract_plate_from_xml(xml_path)
        if not plate:
            failed += 1
            continue

        name_counter[plate] += 1
        if name_counter[plate] == 1:
            new_name = f"{plate}{img_path.suffix.lower()}"
        else:
            new_name = f"{plate}_{name_counter[plate]}{img_path.suffix.lower()}"

        dest = output_folder / new_name
        shutil.copy2(img_path, dest)
        success += 1

    return success, failed

def create_yolo_label(img_path: Path, label_path: Path) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("0 0.500000 0.500000 1.000000 1.000000\n")

def prepare_plate_dataset(
    train_src: Path,
    val_src: Path,
    test_src: Path | None = None,
    output_root: Path = Path("data/plate_dataset"),
) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)

    images_train = output_root / "images" / "train"
    images_val = output_root / "images" / "val"
    labels_train = output_root / "labels" / "train"
    labels_val = output_root / "labels" / "val"

    for d in [images_train, images_val, labels_train, labels_val]:
        d.mkdir(parents=True, exist_ok=True)

    def process(src: Path, img_dst: Path, lbl_dst: Path) -> int:
        count = 0
        for img_path in src.glob("*.*"):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            shutil.copy2(img_path, img_dst / img_path.name)
            create_yolo_label(img_path, lbl_dst / (img_path.stem + ".txt"))
            count += 1
        return count

    n_train = process(train_src, images_train, labels_train)
    n_val = process(val_src, images_val, labels_val)
    if test_src is not None:
        n_val += process(test_src, images_val, labels_val)

    yaml_content = f"""path: {output_root.as_posix()}
train: images/train
val: images/val
names:
  0: plate
"""
    (output_root / "data.yaml").write_text(yaml_content, encoding="utf-8")

    print(f"YOLO dataset ready → {output_root}")
    print(f"  Train: {n_train} images")
    print(f"  Val  : {n_val} images")

def prepare_ocr_from_xml(
    train_src: Path,
    val_src: Path,
    test_src: Path | None = None,
    output_root: Path = Path("data/ocr_dataset"),
) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    total_success = 0
    total_failed = 0

    print("Processing TRAIN ...")
    s, f = rename_images_using_xml(train_src, output_root)
    total_success += s
    total_failed += f
    print(f"  OK: {s} | Failed: {f}")

    print("Processing VALIDATION ...")
    s, f = rename_images_using_xml(val_src, output_root)
    total_success += s
    total_failed += f
    print(f"  OK: {s} | Failed: {f}")

    if test_src is not None:
        print("Processing TEST ...")
        s, f = rename_images_using_xml(test_src, output_root)
        total_success += s
        total_failed += f
        print(f"  OK: {s} | Failed: {f}")

    print(f"\nOCR dataset ready → {output_root}")
    print(f"Total success: {total_success} | Total failed: {total_failed}")

if __name__ == "__main__":
    TRAIN_FOLDER = Path(r"/mnt/g/Bistun-kavir/train")   # ←←← اینجا مسیر واقعی خودت را بنویس
    VAL_FOLDER   = Path(r"/mnt/g/Bistun-kavir/validation")
    TEST_FOLDER  = Path(r"/mnt/g/Bistun-kavir/test")     # ←←← اگر داری

    print("1) Preparing YOLO dataset...")
    prepare_plate_dataset(
        train_src=TRAIN_FOLDER,
        val_src=VAL_FOLDER,
        test_src=TEST_FOLDER,
        output_root=Path("data/plate_dataset"),
    )

    print("\n2) Preparing OCR dataset (auto-rename with real plate text)...")
    prepare_ocr_from_xml(
        train_src=TRAIN_FOLDER,
        val_src=VAL_FOLDER,
        test_src=TEST_FOLDER,
        output_root=Path("data/ocr_dataset"),
    )

    print("\nDONE!")