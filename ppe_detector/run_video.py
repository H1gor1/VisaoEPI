import sys
import os
import time
import cv2

import config
from detectors import PPEDetector, PersonDetector
from vector_db import MemVecDb, VectorDb
from feature_extractor import Extractor


class WorkerTracker:
    def __init__(self, presence_window=10):
        self.__presence_window = presence_window
        self.__last_seen = {}

    def already_present(self, worker_id: int, current_frame: int):
        last = self.__last_seen.get(worker_id)
        self.__last_seen[worker_id] = current_frame

        return (
            last is not None
            and current_frame - last <= self.__presence_window
        )


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


def summarize_findings(db: VectorDb):
    total_workers = db.total_workers()
    for w_id in range(total_workers):
        print(f"Worker id: {w_id}")
        print(db.get_all_data(w_id))
        print()


def save_frame(
    frame: cv2.typing.MatLike,
    frame_id: int,
    folder: str = "debug_frames",
    prefix: str = "frame",
):
    """
    It's useful for troubleshooting reasons, so we can save the frames and inspect them later
    """
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"{prefix}_{frame_id}.jpg")
    cv2.imwrite(filename, frame)
    print(f"Frame salvo em: {filename}")
    return filename


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
    worker_tracker = WorkerTracker()

    frame_count = 0
    t0 = time.time()
    fps_disp = 0

    print("Executando... 'q' para sair.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # if no person was detect, we can save some perfomance by not passing this frame over the ppe_detector
        if (persons := person_detector.detect(frame)) and (
            ppes := detector.detect(frame)
        ):
            for person in persons:
                px1, py1, px2, py2 = person.bbox
                p_vec = extractor.extract_embedding(frame[py1:py2, px1:px2])
                worker_id = db.create_or_update(p_vec, frame_count)
                in_frame = worker_tracker.already_present(worker_id, frame_count)

                for ppe in ppes:

                    if not ppe.is_violation():
                        continue
                    if not person.contains(ppe):
                        continue

                    if not in_frame:
                        count_violation_key = f"count_{ppe.cls_id}"
                        db.update(
                            worker_id,
                            {
                                count_violation_key: db.get_data(
                                    worker_id, count_violation_key, 0
                                )
                                + 1
                            },
                        )

                    color = (
                        config.COLOR_VIOLATION
                        if ppe.is_violation()
                        else config.COLOR_SAFE
                    )
                    cv2.rectangle(
                        frame,
                        (ppe.bbox.x1, ppe.bbox.y1),
                        (ppe.bbox.x2, ppe.bbox.y2),
                        color,
                        2,
                    )
                    cv2.putText(
                        frame,
                        ppe.cls_name,
                        (ppe.bbox.x1, ppe.bbox.y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"Worker id: {worker_id}",
                        (person.bbox.x1, person.bbox.y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

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

        frame_count += 1
        if frame_count % 10 == 0:
            elapsed = time.time() - t0
            fps_disp = 10 / elapsed if elapsed > 0 else 0
            t0 = time.time()

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n{frame_count} frames. Salvo em: {config.VIDEO_OUTPUT_PATH}")
    summarize_findings(db)


if __name__ == "__main__":
    main()
