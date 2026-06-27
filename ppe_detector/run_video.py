import sys
from typing import DefaultDict
import os
import time
import cv2

import config
from detectors import PPEDetector, PersonDetector
from vector_db import MemVecDb
from feature_extractor import Extractor


def load_stream():
    source = sys.argv[1] if len(sys.argv) > 1 else "webcam"

    if source == "webcam":
        cap = cv2.VideoCapture(config.CAMERA_ID)
        print("Abrindo webcam...")
    else:
        if not os.path.exists(source):
            print(f"Video nao encontrado: {source}")
            sys.exit(1)
        cap = cv2.VideoCapture(source)
        print(f"Abrindo: {source}")

    if not cap.isOpened():
        print("Erro ao abrir video.")
        sys.exit(1)
    return cap


frame_id_mapping: DefaultDict[int, int] = DefaultDict(lambda: 0)

def process_frame(
    frame_id: int,
    frame: cv2.typing.MatLike,
    db: MemVecDb,
    detector: PPEDetector,
    person_detector: PersonDetector,
    extractor: Extractor
):
    detections = person_detector.detect(frame)

    for person_crop in detections:
        ppe_detections = detector.detect(person_crop.image)
        p_vec = extractor.extract_embedding(person_crop.image)
        worker_id = db.create_or_update(p_vec, frame_id)
        in_frame = frame_id_mapping[worker_id] == frame_id-1
        frame_id_mapping[worker_id]=frame_id


        for d in ppe_detections:
            x1, y1, x2, y2 = person_crop.bbox
            x1c, y1c, x2c, y2c = d["bbox"]
            is_viol = d["class_id"] in config.VIOLATION_CLASSES
            color = config.COLOR_VIOLATION if is_viol else config.COLOR_SAFE
            if is_viol and in_frame:
                count_violation_key = f"count_{d["class_id"]}"
                db.update(
                    worker_id,
                    {
                        count_violation_key: db.get_data(worker_id, count_violation_key, 0) + 1
                    }
                )
            else:
                color = config.COLOR_SAFE
            cv2.rectangle(frame, (x1+x1c, y1+y1c), (x2+x2c, y2+y2c), color, 2)
            cv2.putText(
                frame,
                d["class_name"],
                (x1+x1c,  y1+y1c - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )


def main():
    cap = load_stream()

    fps = cap.get(cv2.CAP_PROP_FPS) or config.VIDEO_FPS
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        config.VIDEO_OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (w, h),
    )

    db = MemVecDb()
    detector = PPEDetector(
        config.MODEL_PATH, config.CONFIDENCE_THRESHOLD, config.IOU_THRESHOLD
    )
    person_detector = PersonDetector()
    extractor = Extractor()

    frame_count = 0
    t0 = time.time()
    fps_disp = 0

    print("Executando... 'q' para sair.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break


        process_frame(frame_count, frame, db, detector, person_detector, extractor)
        out.write(frame)
        cv2.putText(
            frame,
            f"FPS: {fps_disp:.1f}",
            (w - 130, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.imshow("PPE Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        frame_count+=1
        if frame_count % 10 == 0:
            elapsed = time.time() - t0
            fps_disp = 10 / elapsed if elapsed > 0 else 0
            t0 = time.time()

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n{frame_count} frames. Salvo em: {config.VIDEO_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
