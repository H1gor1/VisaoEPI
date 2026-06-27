import os

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO
from ultralytics.engine.results import Results

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
YOLO_WEIGHTS = os.path.join(MODEL_DIR, "yolov8n.pt")

PERSON_CLASS = 0
CONFIDENCE_THRESHOLD = 0.6

BoundingBox = tuple[int, int, int, int]

@dataclass(frozen=True)
class PersonCrop:
    image: NDArray[np.uint8]
    bbox: tuple[int, int, int, int]


class PersonDetector:

    def __init__(self):
        os.makedirs(MODEL_DIR, exist_ok=True)

        if not os.path.exists(YOLO_WEIGHTS):
            print("[YOLO] Downloading yolov8n.pt...")

        self._model = YOLO(YOLO_WEIGHTS)

    def detect(
        self,
        frame: NDArray[np.uint8],
    ) -> list[PersonCrop]:

        results: Results = self._model(
            frame,
            verbose=False,
            classes=[PERSON_CLASS],
            conf=CONFIDENCE_THRESHOLD,
        )[0]

        people = []

        if results.boxes is not None:
            h, w = frame.shape[:2]

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # clamp to image bounds (prevents invalid indexing)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                crop = frame[y1:y2, x1:x2]
                people.append(
                    PersonCrop(
                        crop, (x1, y1, x2, y2)
                    )
                )

        return people
