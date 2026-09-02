from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _count_images(folder: Path) -> int:
    return sum(1 for p in folder.glob("*.*") if p.suffix.lower() in IMG_EXTS)


def train(
    config_path: str | Path = "configs/config.yaml",
    dataset_root: str | Path = "data/plate_dataset",
) -> None:
    config_path = Path(config_path).resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg.get("training", {}).get("detector", {})
    data_yaml = (Path(dataset_root) / "data.yaml").resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"{data_yaml} not found. Run prepare_dataset first.")

    train_count = _count_images(data_yaml.parent / "images" / "train")
    val_count = _count_images(data_yaml.parent / "images" / "val")
    print(f"Train images : {train_count}")
    print(f"Val images   : {val_count}")
    if train_count == 0:
        raise FileNotFoundError("No images found in the train folder.")
    if val_count == 0:
        raise FileNotFoundError("No images found in the val folder; training would be unvalidated.")

    # Rewrite `path` to an absolute location so the run works from any cwd.
    data_cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    data_cfg["path"] = data_yaml.parent.as_posix()
    data_yaml.write_text(
        yaml.dump(data_cfg, allow_unicode=True, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )

    device = "cuda" if cfg.get("device") == "cuda" else "cpu"
    model = YOLO(train_cfg.get("base_model", "yolov8s.pt"))

    model.train(
        data=str(data_yaml),
        epochs=int(train_cfg.get("epochs", 100)),
        imgsz=int(train_cfg.get("img_size", cfg["detector"]["img_size"])),
        batch=int(train_cfg.get("batch", 16)),
        device=device,
        project="runs/detect",
        name="iran_plate",
        exist_ok=True,
        patience=int(train_cfg.get("patience", 25)),
        optimizer="AdamW",
        lr0=float(train_cfg.get("lr0", 1e-3)),
        workers=int(train_cfg.get("workers", 4)),
        # Geometry: trucks approach the gate at an angle and the camera is fixed.
        degrees=15.0,
        shear=5.0,
        perspective=0.0005,
        translate=0.1,
        scale=0.5,
        # A plate is never mirrored. Ultralytics flips horizontally 50% of the
        # time by default, which teaches the detector a shape that cannot occur.
        fliplr=0.0,
        flipud=0.0,
        # Photometric: night shifts, dust haze and direct sun on the plate.
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.5,
        mosaic=1.0,
        mixup=0.1,
        close_mosaic=10,
    )

    best = Path("runs/detect/iran_plate/weights/best.pt")
    target = Path(cfg["detector"]["model_path"]).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    if not best.exists():
        print("Training finished but best.pt was not found.")
        return

    # copy2, not replace: replace() fails across filesystems and would also strip
    # the run directory of the very checkpoint you may want to resume from.
    shutil.copy2(best, target)
    print(f"\nDetector training finished. Weights copied to: {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--data", type=str, default="data/plate_dataset")
    args = parser.parse_args()
    train(args.config, args.data)
