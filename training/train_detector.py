from __future__ import annotations

from pathlib import Path

import yaml
from ultralytics import YOLO


def train(config_path: str | Path = "configs/config.yaml") -> None:
    config_path = Path(config_path).resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_yaml = (Path.cwd() / "data" / "plate_dataset" / "data.yaml").resolve()

    if not data_yaml.exists():
        raise FileNotFoundError("data.yaml not found. Run prepare_dataset first.")

    images_train = data_yaml.parent / "images" / "train"
    images_val = data_yaml.parent / "images" / "val"

    train_count = len(list(images_train.glob("*.*")))
    val_count = len(list(images_val.glob("*.*")))

    print(f"Train images : {train_count}")
    print(f"Val images   : {val_count}")

    if train_count == 0:
        raise FileNotFoundError("No images found in train folder.")

    data_cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    data_cfg["path"] = str(data_yaml.parent.resolve())
    data_yaml.write_text(
        yaml.dump(data_cfg, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )

    model = YOLO("yolov8s.pt")

    model.train(
        data=str(data_yaml),
        epochs=100,
        imgsz=640,
        batch=16,
        device="cuda",
        project="runs/detect",
        name="iran_plate",
        exist_ok=True,
        patience=25,
        optimizer="AdamW",
        lr0=0.001,
        augment=True,
        degrees=15.0,
        shear=5.0,
        perspective=0.0005,
        mosaic=1.0,
        mixup=0.1,
        workers=4,
    )

    best = Path("runs/detect/iran_plate/weights/best.pt")
    target = Path(cfg["detector"]["model_path"]).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    if best.exists():
        best.replace(target)
        print(f"\nDetector training finished. Weights saved to: {target}")
    else:
        print("Training finished but best.pt was not found.")


if __name__ == "__main__":
    train()
