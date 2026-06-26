import torch
from ultralytics import YOLO


class PPEDetector:

    def using_gpu(self):
        return self.device == "cuda"

    def __init__(self, model_path, confidence=0.5, iou_thres=0.45, device=None):
        self.confidence = confidence
        self.iou_thres = iou_thres
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[detector] Carregando: {model_path}")
        print(f"[detector] Dispositivo: {self.device}")

        self.model = YOLO(model_path)
        self.names = self.model.names
        print(f"[detector] Classes: {self.names}")

    def detect(self, image):
        results = self.model(
            image,
            conf=self.confidence,
            iou=self.iou_thres,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                clss = result.boxes.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), conf, cls_id in zip(boxes, confs, clss):
                    detections.append({
                        "class_id": int(cls_id),
                        "class_name": self.names[int(cls_id)],
                        "confidence": float(conf),
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    })

        return detections
