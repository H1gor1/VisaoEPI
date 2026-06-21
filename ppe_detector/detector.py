import os
import sys
import torch
import numpy as np

_YOLOV5_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo_repo")
_BASE_DIR = os.path.dirname(os.path.dirname(_YOLOV5_DIR))

if _YOLOV5_DIR not in sys.path:
    sys.path.insert(0, _YOLOV5_DIR)

if not os.path.isdir(_YOLOV5_DIR):
    raise FileNotFoundError(
        f"YOLOv5 nao encontrado em {_YOLOV5_DIR}.\n"
        f"Rode primeiro: python ppe_detector/train/train.py"
    )

from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes


def _resolve_weights(filename):
    candidates = [
        filename,
        os.path.join(_YOLOV5_DIR, filename),
        os.path.join(os.path.dirname(_YOLOV5_DIR), filename),
        os.path.join(_BASE_DIR, filename),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    from utils.general import attempt_download
    return attempt_download(filename)


class PPEDetector:

    def using_gpu(self):
        return self.device == "cuda"

    def __init__(self, model_path, confidence=0.5, iou_thres=0.45, device=None):
        self.confidence = confidence
        self.iou_thres = iou_thres
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.stride = 32
        self.img_size = 640

        full_path = _resolve_weights(model_path)

        print(f"[detector] Carregando: {full_path}")
        print(f"[detector] Dispositivo: {self.device}")

        self.model = attempt_load(full_path, device=self.device, inplace=True, fuse=True)
        self.model.eval()
        self.stride = int(self.model.stride.max())
        self.names = self.model.names
        print(f"[detector] Classes: {self.names}")

    def detect(self, image):
        im0 = image.copy()
        im = letterbox(im0, self.img_size, stride=self.stride, auto=True)[0]
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.device).float() / 255.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)

        with torch.no_grad():
            pred = self.model(im)[0]

        pred = non_max_suppression(
            pred, self.confidence, self.iou_thres, None, False, max_det=1000
        )

        detections = []
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls_id in det.cpu().numpy():
                    x1, y1, x2, y2 = map(int, xyxy)
                    detections.append({
                        "class_id": int(cls_id),
                        "class_name": self.names[int(cls_id)],
                        "confidence": float(conf),
                        "bbox": (x1, y1, x2, y2),
                    })

        return detections
