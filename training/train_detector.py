from __future__ import annotations

from pathlib import Path

import yaml
from ultralytics import YOLO


def train(config_path: str | Path = "configs/config.yaml") -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = YOLO("yolov8s.pt")
    model.train(
        data="data/plate_dataset/data.yaml",
        epochs=100,
        imgsz=cfg["detector"]["img_size"],
        batch=16,
        device=0 if cfg.get("device") == "cuda" else "cpu",
        project="runs/detect",
        name="iran_plate",
        exist_ok=True,
        patience=20,
        optimizer="AdamW",
        lr0=0.001,
        augment=True,
        degrees=15.0,
        shear=5.0,
        perspective=0.0005,
        mosaic=1.0,
        mixup=0.1,
    )
    best = Path("runs/detect/iran_plate/weights/best.pt")
    target = Path(cfg["detector"]["model_path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    best.replace(target)
    print(f"Best weights saved to {target}")


if __name__ == "__main__":
    train()
