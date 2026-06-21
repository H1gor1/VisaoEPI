import sys
import os
import time
import cv2
import numpy as np

import config
from detector import PPEDetector

THRESHOLD=15
MIN_PIXEL_RATIO=0.003

class PreProcessing:
    def __init__(self, ppe_detector: PPEDetector):
        self.__detector = ppe_detector
        self.__last_frame = None
        self.__last_detections = None

    def __has_significant_change(self, frame):
        step=8
        small_frame = frame[::step, ::step]                            # view, ~32K pixels
        current_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)   # only on the small frame
        current_gray = cv2.GaussianBlur(current_gray, (5, 5), 0)       # only on the small frame
        if self.__last_frame is None:
            self.__last_frame = current_gray
            return True
    
        prev_gray = self.__last_frame
        
        diff = cv2.absdiff(current_gray, prev_gray)
        self.__last_frame = current_gray
        _, thresh = cv2.threshold(diff, THRESHOLD, 255, cv2.THRESH_BINARY)

        return (np.count_nonzero(thresh) / thresh.size) > MIN_PIXEL_RATIO
    
    def detect(self, frame):
        if self.__has_significant_change(frame):
            self.__last_detections = self.__detector.detect(frame)
        return self.__last_detections


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else "webcam"

    if source == "webcam":
        cap = cv2.VideoCapture(config.CAMERA_ID)
        print("Abrindo webcam...")
    else:
        if not os.path.exists(source):
            print(f"Video nao encontrado: {source}")
            return
        cap = cv2.VideoCapture(source)
        print(f"Abrindo: {source}")

    if not cap.isOpened():
        print("Erro ao abrir video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or config.VIDEO_FPS
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        config.VIDEO_OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h),
    )

    detector = PPEDetector(config.MODEL_PATH, config.CONFIDENCE_THRESHOLD, config.IOU_THRESHOLD)
    if not detector.using_gpu():
        detector = PreProcessing(detector)

    frame_count = 0
    t0 = time.time()
    fps_disp = 0

    print("Executando... 'q' para sair.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        last_detections = detector.detect(frame)
        
        for d in last_detections:
            x1, y1, x2, y2 = d["bbox"]
            is_viol = d["class_id"] in config.VIOLATION_CLASSES
            color = config.COLOR_VIOLATION if is_viol else config.COLOR_SAFE
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, d["class_name"], (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        frame_count += 1
        if frame_count % 10 == 0:
            elapsed = time.time() - t0
            fps_disp = 10 / elapsed if elapsed > 0 else 0
            t0 = time.time()

        cv2.putText(frame, f"FPS: {fps_disp:.1f}", (w - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)
        cv2.imshow("PPE Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n{frame_count} frames. Salvo em: {config.VIDEO_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
