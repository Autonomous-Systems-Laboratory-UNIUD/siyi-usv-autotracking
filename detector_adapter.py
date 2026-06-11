from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
from numpy.typing import NDArray


TARGET_LABEL = None


@dataclass(slots=True, frozen=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    label: str

    @property
    def center(self):
        return (
            (self.x1 + self.x2) * 0.5,
            (self.y1 + self.y2) * 0.5,
        )


class DetectorAdapter:
    def __init__(
        self,
        model_path: str,
        confidence: float,
        *,
        target_label: str | None = TARGET_LABEL,
        device: str | None = None,
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.target_label = target_label
        self.device = device

        self._model: Any | None = None

    def detect(self, frame: NDArray[np.uint8]) -> Detection | None:
        model = self._get_model()

        result = model.predict(frame, conf=self.confidence, verbose=False)[0]

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return None

        names = result.names

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy()

        h, w = frame.shape[:2]

        best = None

        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
            label = self._resolve_label(names, int(cls_id))

            if self.target_label and label != self.target_label:
                continue

            det = Detection(
                x1=self._clip(int(x1), 0, w - 1),
                y1=self._clip(int(y1), 0, h - 1),
                x2=self._clip(int(x2), 0, w - 1),
                y2=self._clip(int(y2), 0, h - 1),
                confidence=float(conf),
                label=label,
            )

            if best is None or det.confidence > best.confidence:
                best = det

        return best

    def _get_model(self):
        if self._model is not None:
            return self._model

        from ultralytics import YOLO
        import torch

        model = YOLO(self.model_path)
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        self._model = model
        return model

    @staticmethod
    def _resolve_label(names, cls_id: int):
        if isinstance(names, dict):
            return str(names.get(cls_id, cls_id))
        if isinstance(names, (list, tuple)):
            return str(names[cls_id])
        return str(cls_id)

    @staticmethod
    def _clip(v: int, lo: int, hi: int):
        return max(lo, min(v, hi))


class DetectionService:
    def __init__(self):
        self._detector = None
        self._config = None

    def detect(self, model_path, frame, confidence, target_label=None, device=None):
        config = (model_path, confidence, target_label, device)

        if self._detector is None or self._config != config:
            self._detector = DetectorAdapter(
                model_path,
                confidence,
                target_label=target_label,
                device=device,
            )
            self._config = config

        return self._detector.detect(frame)