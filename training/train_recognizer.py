from __future__ import annotations

import argparse
import json
import random
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from models.recognizer import IMAGENET_MEAN, IMAGENET_STD, ResNetCRNN, letterbox_plate
from utils.image_processing import imread_unicode
from utils.plate_utils import normalize_iran_plate, label_from_filename

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


def split_by_plate(
    samples: Sequence[Tuple[Path, str]], val_split: float, seed: int = 1337
) -> Tuple[List[Tuple[Path, str]], List[Tuple[Path, str]]]:
    """Hold out whole plates, never individual files.

    prepare_dataset writes repeat sightings of one plate as `<plate>_2.jpg`, so a
    file-level split drops near-identical photos of the same vehicle into both
    train and validation. The model then scores itself on pictures it has
    effectively memorised and the reported CER comes out better than the truth.
    """
    if not 0 < val_split < 1:
        raise ValueError("val_split must be between 0 and 1")
    by_plate: Dict[str, List[Tuple[Path, str]]] = defaultdict(list)
    for item in samples:
        by_plate[item[1]].append(item)

    plates = sorted(by_plate)
    random.Random(seed).shuffle(plates)

    target = len(samples) * val_split
    val: List[Tuple[Path, str]] = []
    val_plates = set()
    for plate in plates[:-1]:
        if len(val) >= target:
            break
        val.extend(by_plate[plate])
        val_plates.add(plate)

    train = [item for plate in plates if plate not in val_plates for item in by_plate[plate]]
    return train, val


def letter_balanced_weights(
    samples: Sequence[Tuple[Path, str]], strength: float = 1.0, boost: str = ""
) -> List[float]:
    """Per-sample weights that even out the plate-letter distribution.

    Letters are wildly unbalanced in IR-LPR -- 'د' has ~2550 plates while 'ع' has
    ~600 -- and a CTC model handles that by learning to guess the common letters,
    because statistically that pays. Sampling inversely to letter frequency makes
    each letter appear about equally often per epoch.

    `strength` interpolates: 0.0 leaves the natural distribution alone, 1.0 fully
    balances. `boost` doubles the weight of specific letters again, for a site
    that only ever sees one plate class.
    """
    counts: Dict[str, int] = defaultdict(int)
    for _, label in samples:
        counts[plate_letter(label)] += 1

    weights: List[float] = []
    for _, label in samples:
        letter = plate_letter(label)
        w = (1.0 / counts[letter]) ** strength if counts[letter] else 1.0
        if boost and letter and letter in boost:
            w *= 2.0
        weights.append(w)
    return weights


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
def plate_letter(text: str) -> str:
    """The single non-digit character an Iranian plate carries, if present."""
    for ch in text:
        if not ch.isdigit():
            return ch
    return ""


@torch.inference_mode()
def evaluate(model, loader, charset: str, device) -> Tuple[float, float, Dict[str, Tuple[int, int]]]:
    """Return (character error rate, exact-match accuracy, per-letter tally).

    The per-letter tally is (correct, total) keyed by the plate's true letter.
    Overall CER hides the failure that actually matters at a single site: a
    letter with 2% of the training mass can be read wrong every single time
    while the headline number still looks respectable.
    """
    was_training = model.training
    model.eval()
    total_chars = total_dist = correct = seen = 0
    per_letter: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
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
            hit = int(pred == truth)
            correct += hit
            seen += 1
            letter = plate_letter(truth)
            if letter:
                per_letter[letter][0] += int(plate_letter(pred) == letter)
                per_letter[letter][1] += 1
    model.train(was_training)
    tally = {k: (v[0], v[1]) for k, v in per_letter.items()}
    return (total_dist / max(total_chars, 1)), (correct / max(seen, 1)), tally


def format_letter_report(tally: Dict[str, Tuple[int, int]], focus: str = "") -> str:
    """One line per letter: how often the letter itself was read correctly."""
    if not tally:
        return "  (no letters in validation set)"
    lines = []
    for letter, (hit, total) in sorted(tally.items(), key=lambda kv: -kv[1][1]):
        mark = "  <-- site letter" if focus and letter in focus else ""
        lines.append(f"    {letter}  {hit:4d}/{total:<4d}  {hit / max(total, 1):6.1%}{mark}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train(config_path: str = "configs/config.yaml", data_root: str = "data/ocr_dataset") -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rec_cfg = cfg["recognizer"]
    # Two different things that were wrongly the same setting:
    #   recognizer.allowed_letters  - an INFERENCE mask, unsafe in production
    #   training.recognizer.boost_letters - which letters to over-sample while
    #     training, so the model genuinely learns them
    # A site can emphasise its letter during training and still deploy unmasked.
    train_cfg = cfg.get("training", {}).get("recognizer", {})
    focus_letters = (
        train_cfg.get("boost_letters") or rec_cfg.get("allowed_letters", "") or ""
    )
    charset = rec_cfg["charset"]
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device") == "cuda" else "cpu")

    epochs = int(train_cfg.get("epochs", 60))
    batch_size = int(train_cfg.get("batch", 32))
    workers = int(train_cfg.get("workers", 4))
    if epochs < 1 or batch_size < 1 or workers < 0:
        raise ValueError("epochs and batch must be positive; workers must be nonnegative")

    data_path = Path(data_root)
    samples = scan_samples(data_path / "train" if (data_path / "train").is_dir() else data_path, charset)
    if not samples:
        raise FileNotFoundError(f"No usable samples under {data_root}. Run prepare_dataset first.")

    if (data_path / "val").is_dir():
        val_samples = scan_samples(data_path / "val", charset)
        val_labels = {label for _, label in val_samples}
        train_samples = [sample for sample in samples if sample[1] not in val_labels]
        if not train_samples or not val_samples:
            raise ValueError("Empty train/validation split after removing overlapping plates")
    else:
        train_samples, val_samples = split_by_plate(
            samples, float(train_cfg.get("val_split", 0.05)), seed=1337
        )
    if not train_samples or not val_samples:
        raise ValueError("Training requires nonempty train and validation sets with distinct plates")
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)} | device: {device}")

    common = dict(charset=charset, img_height=rec_cfg["img_height"], img_width=rec_cfg["img_width"])
    train_ds = PlateOCRDataset(train_samples, augment=bool(train_cfg.get("augment", True)), **common)
    val_ds = PlateOCRDataset(val_samples, augment=False, **common)

    loader_kwargs = dict(collate_fn=collate_fn, num_workers=workers, pin_memory=(device.type == "cuda"))
    if workers > 0:
        loader_kwargs["persistent_workers"] = True
    balance = float(train_cfg.get("letter_balance", 0.0))
    if balance > 0:
        weights = letter_balanced_weights(train_samples, balance, focus_letters)
        sampler = WeightedRandomSampler(weights, num_samples=len(train_samples), replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler, drop_last=False, **loader_kwargs
        )
        print(f"letter-balanced sampling on (strength {balance}"
              + (f", boosting {focus_letters}" if focus_letters else "") + ")")
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=False, **loader_kwargs
        )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    # A fixed, un-augmented slice of the training set, the same size as val.
    # train CER vs val CER is what separates "needs a bigger model" from "needs
    # more data": close together and both high means underfitting, far apart
    # means overfitting, and only the first is fixed by more capacity.
    probe = random.Random(4242).sample(train_samples, min(len(val_samples), len(train_samples)))
    train_probe_ds = PlateOCRDataset(probe, augment=False, **common)
    train_probe_loader = DataLoader(
        train_probe_ds, batch_size=batch_size, shuffle=False, **loader_kwargs
    )

    model = ResNetCRNN(
        num_classes=len(charset) + 1,
        backbone=rec_cfg.get("backbone", "resnet18"),
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
    last_tally: Dict[str, Tuple[int, int]] = {}
    history = []
    manifest = {"train": [[str(p.resolve()), label] for p, label in train_samples],
                "val": [[str(p.resolve()), label] for p, label in val_samples]}
    target_path.with_suffix(".split.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    best_cer = float("inf")
    stale_epochs = 0
    patience = int(train_cfg.get("patience", 12))
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
                longest = max(len(lbl) + sum(a == b for a, b in zip(lbl, lbl[1:])) for _, lbl in train_samples)
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
        cer, acc, tally = evaluate(model, val_loader, charset, device)
        train_cer, _, _ = evaluate(model, train_probe_loader, charset, device)
        flag = ""
        if cer < best_cer:
            # Checkpoint on validation, not at the final epoch: the last epoch is
            # rarely the best one, and the old script only ever saved that.
            best_cer = cer
            torch.save({"model": model.state_dict(), "charset": charset,
                        "backbone": rec_cfg.get("backbone", "resnet18"),
                        "img_height": rec_cfg["img_height"], "img_width": rec_cfg["img_width"],
                        "epoch": epoch, "val_cer": cer, "train_cer": train_cer}, target_path)
            stale_epochs = 0
            flag = "  <- saved"

        site_note = ""
        if focus_letters:
            hit = sum(t[0] for l, t in tally.items() if l in focus_letters)
            tot = sum(t[1] for l, t in tally.items() if l in focus_letters)
            if tot:
                site_note = f" | {focus_letters} acc {hit / tot:.3f}"
        print(
            f"Epoch {epoch:03d} | loss {avg_loss:.4f} | train CER {train_cer:.4f} | val CER {cer:.4f} "
            f"| plate acc {acc:.4f}{site_note}{flag}"
        )
        history.append({"epoch": epoch, "loss": avg_loss, "train_cer": train_cer,
                        "val_cer": cer, "plate_accuracy": acc})
        target_path.with_suffix(".history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8")
        last_tally = tally
        if not flag:
            stale_epochs += 1
        if patience > 0 and stale_epochs >= patience:
            print(f"Early stopping after {patience} epochs without validation improvement")
            break

    print(f"\nBest val CER: {best_cer:.4f}")
    gap = cer - train_cer
    verdict = (
        "high training error; inspect optimization and labels before changing model size"
        if train_cer > 0.05 and gap < 0.05
        else "large generalization gap; investigate overfitting, labels and domain shift"
        if gap > 0.10
        else "no large gap detected by this heuristic; not proof against overfitting"
    )
    print(f"Final train CER {train_cer:.4f} vs val CER {cer:.4f}  (gap {gap:+.4f}) -> {verdict}")
    print("\nPer-letter accuracy on the final epoch:")
    print(format_letter_report(last_tally, focus_letters))
    print(f"\nRecognizer weights saved to {target_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--data", type=str, default="data/ocr_dataset")
    args = parser.parse_args()
    train(args.config, args.data)
