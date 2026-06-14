# PPE Detector

Detecção de Equipamentos de Proteção Individual (EPIs) com YOLOv5 e OpenCV.

Baseado no artigo *Nath et al. (2020) — "Deep learning for site safety: Real-time detection of personal protective equipment"*, Automation in Construction.

## Dataset

8 classes do [Kaggle](https://www.kaggle.com/datasets/anuragraj03/ppe-detection-m):

| Id | Classe | Significado |
|----|--------|-------------|
| 0 | no-safety-glove | sem luvas |
| 1 | no-safety-helmet | sem capacete |
| 2 | no-safety-shoes | sem calçado |
| 3 | no-welding-glass | sem óculos de solda |
| 4 | safety-glove | com luvas |
| 5 | safety-helmet | com capacete |
| 6 | safety-shoes | com calçado |
| 7 | welding-glass | com óculos de solda |

## Instalação

```bash
git clone https://github.com/H1gor1/VisaoEPI.git
cd VisaoEPI
pip install -r requirements.txt
```

## Treinamento

```bash
python ppe_detector/train/train.py
```

Parâmetros opcionais:

```bash
python ppe_detector/train/train.py --epochs 100 --batch 8 --weights yolov5m.pt
```

| Argumento | Padrão | Descrição |
|-----------|--------|-----------|
| `--epochs` | 50 | Épocas de treino |
| `--batch` | 16 | Tamanho do batch (reduza se faltar VRAM) |
| `--img` | 640 | Resolução da imagem |
| `--weights` | yolov5s.pt | Modelo base (s=small, m=medium, l=large) |

O script baixa o dataset do Kaggle automaticamente, clona o YOLOv5, gera o `dataset.yaml` e treina.
O modelo final é salvo em `ppe_detector/models/best.pt`.

## Execução

### Imagem

```bash
python ppe_detector/run_image.py                       # usa test_images/
python ppe_detector/run_image.py caminho/da/foto.jpg   # imagem específica
```

### Vídeo / Webcam

```bash
python ppe_detector/run_video.py            # webcam
python ppe_detector/run_video.py video.mp4  # arquivo
```

Pressione `q` para sair do vídeo.

### Ajuste de thresholds

Edite `ppe_detector/config.py`:

```python
CONFIDENCE_THRESHOLD = 0.25   # confiança mínima (menor = mais detecções)
IOU_THRESHOLD = 0.45          # sobreposição máxima para NMS
```

## Estrutura

```
ppe_detector/
├── config.py           # paths, classes, thresholds, cores
├── detector.py         # PPEDetector (inferência YOLOv5)
├── run_image.py        # pipeline para imagem estática
├── run_video.py        # pipeline para vídeo/webcam
├── train/
│   └── train.py        # download dataset + treino YOLOv5
├── models/
│   └── best.pt         # modelo treinado
└── test_images/        # imagens de teste
```

## Resultados do treino

| Métrica | Valor |
|---------|-------|
| Modelo | YOLOv5s |
| mAP@0.5 | 0.701 |
| Épocas | 50 |
| Imagens | ~15K |

Melhores classes: safety-glove (0.88), safety-helmet (0.85).
Piores classes: no-safety-shoes (0.53), no-welding-glass (0.51).
