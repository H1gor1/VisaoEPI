import os

import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO
from ultralytics.engine.results import Results

from .types import Detection, Coords

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
YOLO_WEIGHTS = os.path.join(MODEL_DIR, "yolov8n.pt")

PERSON_CLASS = 0
CONFIDENCE_THRESHOLD = 0.4


class PersonDetector:

    def __init__(self):
        os.makedirs(MODEL_DIR, exist_ok=True)

        if not os.path.exists(YOLO_WEIGHTS):
            print("[YOLO] Downloading yolov8n.pt...")

        self._model = YOLO(YOLO_WEIGHTS)
        self.names = self._model.names

    def detect(
        self,
        frame: NDArray[np.uint8],
    ) -> list[Detection]:

        results: Results = self._model(
            frame,
            verbose=False,
            classes=[PERSON_CLASS],
            conf=CONFIDENCE_THRESHOLD,
        )

        people = []
        for result in results:
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                clss = result.boxes.cls.cpu().numpy().astype(int)

                for (x1, y1, x2, y2), conf, cls_id in zip(boxes, confs, clss):
                    people.append(
                        Detection(
                            cls_id=int(cls_id),
                            cls_name=self.names[int(cls_id)],
                            confidence=float(conf),
                            bbox=Coords(int(x1), int(y1), int(x2), int(y2)),
                        )
                    )

        return people
