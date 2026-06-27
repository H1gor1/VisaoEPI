import sys
import os
import cv2

import config
from detectors import PPEDetector


def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        if not os.path.isdir(config.TEST_IMAGES_DIR):
            print("Forneca uma imagem: python run_image.py <caminho>")
            return
        images = sorted(f for f in os.listdir(config.TEST_IMAGES_DIR)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if not images:
            print("Nenhuma imagem em test_images/")
            return
        image_path = os.path.join(config.TEST_IMAGES_DIR, images[0])

    if not os.path.exists(image_path):
        print(f"Imagem nao encontrada: {image_path}")
        return

    print(f"Carregando: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print("Erro ao carregar imagem.")
        return

    detector = PPEDetector(config.MODEL_PATH, config.CONFIDENCE_THRESHOLD, config.IOU_THRESHOLD)
    detections = detector.detect(frame)

    for d in detections:
        x1, y1, x2, y2 = d.bbox
        is_viol = d.cls_id in config.VIOLATION_CLASSES
        color = config.COLOR_VIOLATION if is_viol else config.COLOR_SAFE
        tag = "VIOLACAO" if is_viol else "OK"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{d.cls_name} {d.confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        print(f"  [{tag}] {d.cls_name}: {d.confidence:.2f}")

    print(f"\n{len(detections)} EPI(s) detectado(s)")

    cv2.imshow("PPE Detector", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("resultado.jpg", frame)
    print("Resultado salvo em: resultado.jpg")


if __name__ == "__main__":
    main()
