# PPE Detector

Deteccao de EPIs com YOLOv5.

## Dataset

Kaggle: anuragraj03/ppe-detection-m (8 classes)

## Instalar

    pip install -r requirements.txt

## Treinar

    python ppe_detector/train/train.py
    python ppe_detector/train/train.py --epochs 100 --batch 8 --weights yolov5m.pt

## Testar

    python ppe_detector/run_image.py
    python ppe_detector/run_image.py foto.jpg

    python ppe_detector/run_video.py
    python ppe_detector/run_video.py video.mp4

## Thresholds

Editar ppe_detector/config.py
