import sys
import os
import time
import cv2

import config
from detector import PPEDetector


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

    frame_count = 0
    t0 = time.time()
    fps_disp = 0

    print("Executando... 'q' para sair.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)

        for d in detections:
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
