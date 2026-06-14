# PPE Detector

Trabalho de Visao Computacional - Deteccao de EPIs com YOLOv5

## Dataset

8 classes do Kaggle (anuragraj03/ppe-detection-m):

0 - no-safety-glove
1 - no-safety-helmet
2 - no-safety-shoes
3 - no-welding-glass
4 - safety-glove
5 - safety-helmet
6 - safety-shoes
7 - welding-glass

## Como usar

Instalar dependencias:

    pip install -r requirements.txt

Treinar o modelo:

    python ppe_detector/train/train.py

O dataset eh baixado automaticamente. O modelo treinado vai pra ppe_detector/models/best.pt.

Testar com imagem:

    python ppe_detector/run_image.py
    python ppe_detector/run_image.py foto.jpg

Testar com video/webcam:

    python ppe_detector/run_video.py
    python ppe_detector/run_video.py video.mp4

Apertar 'q' pra sair do video.

## Thresholds

Editar ppe_detector/config.py:

    CONFIDENCE_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.45

## Treino com mais epochs ou outro modelo

    python ppe_detector/train/train.py --epochs 100 --batch 8 --weights yolov5m.pt

## Resultados

YOLOv5s, 50 epochs, mAP@0.5 = 0.701
