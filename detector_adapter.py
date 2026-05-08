from __future__ import annotations

from dataclasses import dataclass
from numpy import ndarray


TARGET_LABEL = None


@dataclass(slots=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 1.0
    label: str = "target"

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


class DetectorAdapter:
    def __init__(
        self,
        model_path: str,
        confidence: float,
        *,
        target_label: str | None = TARGET_LABEL,
        device: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self.target_label = target_label
        self.device = device
        self._model = None
        self._load_error = None

    def detect(self, frame: ndarray[tuple[int, int, int]]) -> Detection | None:
        model = self._load_model()
        results = model.predict(frame, conf=self.confidence, verbose=False)[0]

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return None

        names = results.names
        xyxy = boxes.xyxy.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        classes = boxes.cls.cpu().tolist()

        best_detection = None
        height, width = frame.shape[:2]

        for coords, confidence, cls_value in zip(xyxy, confidences, classes):
            cls_id = int(cls_value)
            label = _label_from_names(names, cls_id)
            if self.target_label is not None and label != self.target_label:
                continue

            x1, y1, x2, y2 = (int(value) for value in coords)
            x1 = max(0, min(width - 1, x1))
            x2 = max(0, min(width - 1, x2))
            y1 = max(0, min(height - 1, y1))
            y2 = max(0, min(height - 1, y2))

            candidate = Detection(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=confidence,
                label=label,
            )
            if (
                best_detection is None
                or candidate.confidence > best_detection.confidence
            ):
                best_detection = candidate

        return best_detection

    def _load_model(self):
        if self._model is not None:
            return self._model

        if self._load_error is not None:
            raise self._load_error

        try:
            import torch
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path))
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(device)
            return self._model
        except Exception as exc:
            self._load_error = RuntimeError(
                f"Failed to load model at {self.model_path}: {exc}"
            )
            raise self._load_error from exc


def _label_from_names(names, cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


_default_detector = None
_default_detector_config = None


def detect_target(
    model_path: str,
    frame: ndarray[tuple[int, int, int]],
    confidence: float,
) -> Detection | None:
    global _default_detector, _default_detector_config

    config = (model_path, confidence, TARGET_LABEL)
    if _default_detector is None or _default_detector_config != config:
        _default_detector = DetectorAdapter(
            model_path,
            confidence,
            target_label=TARGET_LABEL,
        )
        _default_detector_config = config

    return _default_detector.detect(frame)
