from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import yaml

from models.recognizer import ResNetCRNN


class PlateOCRDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        charset: str,
        img_height: int = 64,
        img_width: int = 256,
    ) -> None:
        self.root = Path(root)
        self.charset = charset
        self.char_to_idx = {c: i for i, c in enumerate(charset)}
        self.blank_idx = len(charset)
        self.img_height = img_height
        self.img_width = img_width
        self.samples: List[Tuple[Path, str]] = []

        for img_path in self.root.glob("**/*.*"):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            label = img_path.stem
            if all(c in charset for c in label):
                self.samples.append((img_path, label))

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        path, label = self.samples[idx]
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_width, self.img_height), interpolation=cv2.INTER_CUBIC)
        tensor = self.transform(img)

        target = torch.tensor([self.char_to_idx[c] for c in label], dtype=torch.long)
        return tensor, target, len(target)


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]):
    images, targets, lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return images, targets, lengths


def train(config_path: str = "configs/config.yaml") -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rec_cfg = cfg["recognizer"]
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device") == "cuda" else "cpu")

    dataset = PlateOCRDataset(
        root="data/ocr_dataset",
        charset=rec_cfg["charset"],
        img_height=rec_cfg["img_height"],
        img_width=rec_cfg["img_width"],
    )
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model = ResNetCRNN(
        num_classes=len(rec_cfg["charset"]) + 1,
        backbone=rec_cfg.get("backbone", "resnet18"),
        pretrained=rec_cfg.get("pretrained", True),
    ).to(device)

    criterion = nn.CTCLoss(blank=len(rec_cfg["charset"]), zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    model.train()
    for epoch in range(1, 51):
        total_loss = 0.0
        for images, targets, target_lengths in loader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            logits = model(images)
            log_probs = logits.log_softmax(2).permute(1, 0, 2)
            input_lengths = torch.full(
                size=(images.size(0),),
                fill_value=logits.size(1),
                dtype=torch.long,
                device=device,
            )

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / max(len(loader), 1)
        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f}")

    target = Path(rec_cfg["model_path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), target)
    print(f"Recognizer weights saved to {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    train(args.config)
