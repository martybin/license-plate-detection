from __future__ import annotations

import argparse
import random
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset

from models.recognizer import IMAGENET_MEAN, IMAGENET_STD, ResNetCRNN, letterbox_plate
from utils.image_processing import imread_unicode
from utils.plate_utils import normalize_iran_plate

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# --------------------------------------------------------------------------- #
# Augmentation: reproduce the conditions the mine gate actually sees.
# --------------------------------------------------------------------------- #

def _motion_blur(img: np.ndarray, rng: random.Random) -> np.ndarray:
    size = rng.choice([5, 7, 9, 11])
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0
    matrix = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), rng.uniform(0, 180), 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (size, size))
    total = kernel.sum()
    return cv2.filter2D(img, -1, kernel / total) if total > 0 else img


def _dust_haze(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Blend toward an ochre veil, the way airborne mine dust washes out a plate."""
    strength = rng.uniform(0.15, 0.5)
    veil = np.full_like(img, (110, 140, 165), dtype=np.uint8)  # BGR, dusty ochre
    hazed = cv2.addWeighted(img, 1 - strength, veil, strength, 0)

    h, w = img.shape[:2]
    noise = rng.uniform(0.0, 12.0)
    if noise > 1.0:
        low = np.random.default_rng(rng.randrange(1 << 30)).normal(0, noise, (max(2, h // 8), max(2, w // 8)))
        low = cv2.resize(low.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        hazed = np.clip(hazed.astype(np.float32) + low[..., None], 0, 255).astype(np.uint8)
    return hazed


def _sun_glare(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Add a blown-out specular patch, as when the sun hits a reflective plate."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    center = (rng.randrange(w), rng.randrange(h))
    axes = (rng.randint(w // 6, max(w // 6 + 1, w // 2)), rng.randint(h // 4, max(h // 4 + 1, h)))
    cv2.ellipse(mask, center, axes, rng.uniform(0, 180), 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), max(3, min(h, w) / 6)) * rng.uniform(0.4, 0.95)
    return np.clip(img.astype(np.float32) + mask[..., None] * 255.0, 0, 255).astype(np.uint8)


def _night(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Darken and desaturate, then add read noise like a real low-light sensor."""
    gamma = rng.uniform(1.6, 3.2)
    table = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
    dark = cv2.LUT(img, table)
    dark = cv2.addWeighted(dark, rng.uniform(0.75, 1.0), np.zeros_like(dark), 0, rng.uniform(-10, 10))
    noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(4, 16), img.shape)
    return np.clip(dark.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _perspective(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    jitter = rng.uniform(0.02, 0.09)
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst = src + np.array(
        [[rng.uniform(-jitter, jitter) * w, rng.uniform(-jitter, jitter) * h] for _ in range(4)],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _jpeg(img: np.ndarray, rng: random.Random) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), rng.randint(25, 70)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def augment_plate(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply a random subset of the harsh-condition transforms.

    Each is applied independently so the model also sees combinations - dust at
    night, glare on a blurred plate - which is what actually happens on site.
    """
    if rng.random() < 0.35:
        img = _perspective(img, rng)
    if rng.random() < 0.30:
        img = _motion_blur(img, rng)
    elif rng.random() < 0.20:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.6, 2.2))
    if rng.random() < 0.25:
        img = _dust_haze(img, rng)
    if rng.random() < 0.20:
        img = _sun_glare(img, rng)
    if rng.random() < 0.25:
        img = _night(img, rng)
    if rng.random() < 0.30:
        alpha = rng.uniform(0.7, 1.35)
        img = np.clip(img.astype(np.float32) * alpha + rng.uniform(-25, 25), 0, 255).astype(np.uint8)
    if rng.random() < 0.25:
        img = _jpeg(img, rng)
    return img


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

def label_from_filename(path: Path) -> str:
    """Recover the plate text from a filename.

    prepare_dataset writes duplicates as `<plate>_2.jpg`. The old loader used the
    raw stem, so every one of those (2268 of 25835 files) failed the charset
    check and was silently dropped from training.
    """
    stem = path.stem
    if "_" in stem:
        head, _, tail = stem.rpartition("_")
        if head and tail.isdigit():
            stem = head
    return normalize_iran_plate(stem)


class PlateOCRDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Tuple[Path, str]],
        charset: str,
        img_height: int = 64,
        img_width: int = 256,
        augment: bool = False,
        seed: int = 0,
    ) -> None:
        self.samples = list(samples)
        self.char_to_idx = {c: i for i, c in enumerate(charset)}
        self.img_height = img_height
        self.img_width = img_width
        self.augment = augment
        self.seed = seed
        self.read_failures = 0
        self._rng: Optional[random.Random] = None

    def __len__(self) -> int:
        return len(self.samples)

    def _worker_rng(self) -> random.Random:
        """One RNG per worker whose state advances with every sample drawn.

        Deriving the seed from the sample index instead made the augmentation a
        pure function of that index: every epoch produced byte-identical images,
        so 60 epochs saw one fixed dusty/blurred variant per plate rather than a
        fresh one each time. torch.initial_seed() differs per worker, which keeps
        the workers from drawing the same stream.
        """
        if self._rng is None:
            self._rng = random.Random(torch.initial_seed() + self.seed)
        return self._rng

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        path, label = self.samples[idx]
        # Every file here is named after its plate, so the path is non-ASCII and
        # cv2.imread would return None on Windows -- silently training the whole
        # run on blank images.
        img = imread_unicode(path)
        if img is None:
            self.read_failures += 1
            if self.read_failures <= 5:
                warnings.warn(f"Could not read {path}; using a blank image", RuntimeWarning)
            img = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)

        if self.augment:
            img = augment_plate(img, self._worker_rng())

        img = letterbox_plate(img, self.img_height, self.img_width)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD

        tensor = torch.from_numpy(img).permute(2, 0, 1)
        target = torch.tensor([self.char_to_idx[c] for c in label], dtype=torch.long)
        return tensor, target, len(target)


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]):
    images, targets, lengths = zip(*batch)
    return (
        torch.stack(images, 0),
        torch.cat(targets, 0),
        torch.tensor(lengths, dtype=torch.long),
    )


def scan_samples(root: Path, charset: str, min_len: int = 5, max_len: int = 10) -> List[Tuple[Path, str]]:
    samples: List[Tuple[Path, str]] = []
    skipped = 0
    charset_set = set(charset)
    for path in sorted(root.glob("**/*")):
        if path.suffix.lower() not in IMG_EXTS:
            continue
        label = label_from_filename(path)
        if min_len <= len(label) <= max_len and all(c in charset_set for c in label):
            samples.append((path, label))
        else:
            skipped += 1
    print(f"Usable samples: {len(samples)} | skipped: {skipped}")
    return samples


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def greedy_decode(logits: torch.Tensor, charset: str) -> List[str]:
    blank = len(charset)
    preds = logits.argmax(dim=2).cpu().numpy()
    texts = []
    for seq in preds:
        chars, prev = [], -1
        for p in seq:
            if p != prev and p != blank:
                chars.append(charset[int(p)])
            prev = p
        texts.append("".join(chars))
    return texts


@torch.no_grad()
def evaluate(model, loader, charset: str, device) -> Tuple[float, float]:
    """Return (character error rate, exact-match accuracy)."""
    model.eval()
    total_chars = total_dist = correct = seen = 0
    for images, targets, lengths in loader:
        logits = model(images.to(device))
        offset = 0
        truths = []
        for length in lengths.tolist():
            truths.append("".join(charset[i] for i in targets[offset : offset + length].tolist()))
            offset += length
        for pred, truth in zip(greedy_decode(logits, charset), truths):
            total_dist += edit_distance(pred, truth)
            total_chars += len(truth)
            correct += int(pred == truth)
            seen += 1
    model.train()
    return (total_dist / max(total_chars, 1)), (correct / max(seen, 1))


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train(config_path: str = "configs/config.yaml", data_root: str = "data/ocr_dataset") -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rec_cfg = cfg["recognizer"]
    train_cfg = cfg.get("training", {}).get("recognizer", {})
    charset = rec_cfg["charset"]
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device") == "cuda" else "cpu")

    epochs = int(train_cfg.get("epochs", 60))
    batch_size = int(train_cfg.get("batch", 32))
    workers = int(train_cfg.get("workers", 4))

    samples = scan_samples(Path(data_root), charset)
    if not samples:
        raise FileNotFoundError(f"No usable samples under {data_root}. Run prepare_dataset first.")

    rng = random.Random(1337)
    rng.shuffle(samples)
    n_val = max(1, int(len(samples) * float(train_cfg.get("val_split", 0.05))))
    val_samples, train_samples = samples[:n_val], samples[n_val:]
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)} | device: {device}")

    common = dict(charset=charset, img_height=rec_cfg["img_height"], img_width=rec_cfg["img_width"])
    train_ds = PlateOCRDataset(train_samples, augment=bool(train_cfg.get("augment", True)), **common)
    val_ds = PlateOCRDataset(val_samples, augment=False, **common)

    loader_kwargs = dict(collate_fn=collate_fn, num_workers=workers, pin_memory=(device.type == "cuda"))
    if workers > 0:
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    model = ResNetCRNN(
        num_classes=len(charset) + 1,
        backbone=rec_cfg.get("backbone", "resnet34"),
        pretrained=rec_cfg.get("pretrained", True),
    ).to(device)

    criterion = nn.CTCLoss(blank=len(charset), zero_infinity=True)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(train_cfg.get("lr", 3e-4)),
        epochs=epochs,
        steps_per_epoch=max(1, len(train_loader)),
        pct_start=0.1,
    )
    use_amp = device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    except (AttributeError, TypeError):  # torch < 2.4
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    target_path = Path(rec_cfg["model_path"])
    target_path.parent.mkdir(parents=True, exist_ok=True)
    best_cer = float("inf")
    checked_timesteps = False

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for images, targets, target_lengths in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)

            if not checked_timesteps:
                # CTC cannot represent a label longer than the sequence it emits;
                # fail loudly here rather than train for hours toward nothing.
                longest = max(len(lbl) for _, lbl in train_samples)
                if logits.size(1) < longest:
                    raise ValueError(
                        f"Only {logits.size(1)} CTC timesteps for labels up to {longest} chars. "
                        "Increase recognizer.img_width or reduce the backbone stride."
                    )
                print(f"CTC timesteps: {logits.size(1)} (longest label: {longest})")
                checked_timesteps = True

            log_probs = logits.float().log_softmax(2).permute(1, 0, 2)
            input_lengths = torch.full(
                (images.size(0),), logits.size(1), dtype=torch.long, device=device
            )
            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        cer, acc = evaluate(model, val_loader, charset, device)
        flag = ""
        if cer < best_cer:
            # Checkpoint on validation, not at the final epoch: the last epoch is
            # rarely the best one, and the old script only ever saved that.
            best_cer = cer
            torch.save(model.state_dict(), target_path)
            flag = "  <- saved"
        print(f"Epoch {epoch:03d} | loss {avg_loss:.4f} | val CER {cer:.4f} | plate acc {acc:.4f}{flag}")

    print(f"\nBest val CER: {best_cer:.4f}")
    print(f"Recognizer weights saved to {target_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--data", type=str, default="data/ocr_dataset")
    args = parser.parse_args()
    train(args.config, args.data)
