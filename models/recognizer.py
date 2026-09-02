from __future__ import annotations

from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Recognition(NamedTuple):
    """A single OCR reading plus the confidence the network assigned to it."""

    text: str
    confidence: float


def _shrink_width_stride(layer: nn.Module) -> None:
    """Turn a (2, 2) stride into (2, 1) so the width axis stops halving.

    A stock ResNet downsamples by 32, which leaves a 256px wide plate with only
    8 CTC timesteps - fewer than the 8..9 characters an Iranian plate carries,
    so the loss is unreachable by construction. Keeping the height stride but
    freezing the width stride in layer3/layer4 yields 32 timesteps instead.
    """
    for module in layer.modules():
        if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
            module.stride = (2, 1)
        elif isinstance(module, nn.MaxPool2d) and module.stride == 2:
            module.stride = (2, 1)


class ResNetCRNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet34",
        pretrained: bool = True,
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if backbone == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            feat_dim = 512
        elif backbone == "resnet34":
            base = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            feat_dim = 512
        elif backbone == "resnet50":
            base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            feat_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        _shrink_width_stride(base.layer3)
        _shrink_width_stride(base.layer4)

        self.cnn = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn = nn.LSTM(
            input_size=feat_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)
        features = self.avgpool(features)
        features = features.squeeze(2)
        features = features.permute(0, 2, 1)
        rnn_out, _ = self.rnn(features)
        logits = self.fc(self.dropout(rnn_out))
        return logits


def letterbox_plate(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize into the target box while preserving aspect ratio.

    Plate crops arrive at wildly different tightness (90x22 up to 600x300 in the
    training set). Squashing them all into a fixed box distorts glyph shapes by a
    different amount per crop; scaling then edge-padding keeps them consistent.
    """
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    scale = min(width / w, height / h)
    new_w = max(1, min(width, int(round(w * scale))))
    new_h = max(1, min(height, int(round(h * scale))))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    pad_x = width - new_w
    pad_y = height - new_h
    if pad_x or pad_y:
        top = pad_y // 2
        left = pad_x // 2
        resized = cv2.copyMakeBorder(
            resized,
            top,
            pad_y - top,
            left,
            pad_x - left,
            cv2.BORDER_REPLICATE,
        )
    return resized


class PlateRecognizer:
    def __init__(
        self,
        model_path: str | Path,
        charset: str,
        backbone: str = "resnet34",
        img_height: int = 64,
        img_width: int = 256,
        device: str = "cuda",
        half: bool = True,
    ) -> None:
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.charset = charset
        self.idx_to_char = {i: c for i, c in enumerate(charset)}
        self.blank_idx = len(charset)
        self.img_height = img_height
        self.img_width = img_width
        self.use_half = half and self.device == "cuda"

        # pretrained=False on purpose: the ImageNet weights would be overwritten
        # by the checkpoint on the next line anyway, and downloading them would
        # make startup depend on internet access the mine gate may not have.
        self.model = ResNetCRNN(
            num_classes=len(charset) + 1,
            backbone=backbone,
            pretrained=False,
        )
        self.model.load_state_dict(self._load_state(model_path))
        self.model.to(self.device)
        self.model.eval()
        if self.use_half:
            self.model.half()

    def _load_state(self, model_path: str | Path) -> dict:
        try:
            state = torch.load(str(model_path), map_location=self.device, weights_only=True)
        except TypeError:  # torch older than 2.0 has no weights_only
            state = torch.load(str(model_path), map_location=self.device)
        if isinstance(state, dict) and isinstance(state.get("model"), dict):
            state = state["model"]
        return state

    def _preprocess(self, images: Sequence[np.ndarray]) -> torch.Tensor:
        batch = np.empty((len(images), self.img_height, self.img_width, 3), dtype=np.float32)
        for i, image in enumerate(images):
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                # Training reads with cv2.imread then BGR2RGB; match it exactly or
                # every colour cue the network learned is swapped at inference.
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = letterbox_plate(image, self.img_height, self.img_width)
            batch[i] = (image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD

        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device)
        return tensor.half() if self.use_half else tensor

    def _ctc_decode(self, logits: torch.Tensor) -> List[Recognition]:
        probs = F.softmax(logits.float(), dim=2)
        confs, preds = probs.max(dim=2)
        preds = preds.cpu().numpy()
        confs = confs.cpu().numpy()

        readings: List[Recognition] = []
        for seq, seq_conf in zip(preds, confs):
            chars: List[str] = []
            char_confs: List[float] = []
            prev = -1
            for p, c in zip(seq, seq_conf):
                if p != prev and p != self.blank_idx and p < len(self.charset):
                    chars.append(self.idx_to_char[int(p)])
                    char_confs.append(float(c))
                prev = p
            # Geometric mean: one badly-read glyph should drag the score down
            # rather than be averaged away by its confident neighbours.
            confidence = (
                float(np.exp(np.mean(np.log(np.clip(char_confs, 1e-6, 1.0))))) if char_confs else 0.0
            )
            readings.append(Recognition("".join(chars), confidence))
        return readings

    @torch.inference_mode()
    def recognize_batch(self, plate_imgs: Sequence[np.ndarray]) -> List[Optional[Recognition]]:
        results: List[Optional[Recognition]] = [None] * len(plate_imgs)
        valid = [(i, img) for i, img in enumerate(plate_imgs) if img is not None and img.size > 0]
        if not valid:
            return results

        logits = self.model(self._preprocess([img for _, img in valid]))
        for (i, _), reading in zip(valid, self._ctc_decode(logits)):
            results[i] = reading if reading.text else None
        return results

    def recognize(self, plate_img: np.ndarray) -> Optional[Recognition]:
        return self.recognize_batch([plate_img])[0]
