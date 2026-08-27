from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNetCRNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet34",
        pretrained: bool = True,
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
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1,
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)
        features = self.avgpool(features)
        features = features.squeeze(2)
        features = features.permute(0, 2, 1)
        rnn_out, _ = self.rnn(features)
        logits = self.fc(rnn_out)
        return logits


class PlateRecognizer:
    def __init__(
        self,
        model_path: str | Path,
        charset: str,
        backbone: str = "resnet34",
        img_height: int = 64,
        img_width: int = 256,
        device: str = "cuda",
        pretrained: bool = True,
    ) -> None:
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.charset = charset
        self.idx_to_char = {i: c for i, c in enumerate(charset)}
        self.blank_idx = len(charset)
        self.img_height = img_height
        self.img_width = img_width

        self.model = ResNetCRNN(
            num_classes=len(charset) + 1,
            backbone=backbone,
            pretrained=pretrained,
        )
        state = torch.load(str(model_path), map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.resize(image, (self.img_width, self.img_height), interpolation=cv2.INTER_CUBIC)
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def _ctc_decode(self, logits: torch.Tensor) -> str:
        probs = F.softmax(logits, dim=2)
        preds = torch.argmax(probs, dim=2)[0].cpu().numpy()
        chars: List[str] = []
        prev = -1
        for p in preds:
            if p != prev and p != self.blank_idx:
                if p < len(self.charset):
                    chars.append(self.idx_to_char[p])
            prev = p
        return "".join(chars)

    @torch.no_grad()
    def recognize(self, plate_img: np.ndarray) -> Optional[str]:
        if plate_img is None or plate_img.size == 0:
            return None
        tensor = self._preprocess(plate_img)
        logits = self.model(tensor)
        text = self._ctc_decode(logits)
        return text if text else None
