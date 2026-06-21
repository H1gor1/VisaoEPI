import os
import sys
import shutil
import argparse

from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

MODEL_CHOICES = {
    "nano": "yolov8n.pt",
    "small": "yolov8s.pt",
    "medium": "yolov8m.pt",
}

def main():
    parser = argparse.ArgumentParser(description="Treinar YOLOv8 no dataset de EPIs (maquina local)")
    parser.add_argument("--model", type=str, default="nano", choices=list(MODEL_CHOICES),
                        help="Tamanho do modelo YOLOv8")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Numero de epochs")
    parser.add_argument("--batch", type=int, default=8,
                        help="Tamanho do batch (reduza se der OOM)")
    parser.add_argument("--img", type=int, default=640,
                        help="Tamanho da imagem de entrada")
    args = parser.parse_args()

    weights = MODEL_CHOICES[args.model]

    if not os.path.exists(config.DATASET_YAML):
        print(f"[erro] dataset.yaml nao encontrado em: {config.DATASET_YAML}")
        print("[erro] Execute primeiro o train.py (YOLOv5) para baixar o dataset")
        sys.exit(1)

    print("=" * 50)
    print("  PPE Detector - Treinamento YOLOv8 (Local)")
    print("=" * 50)
    print(f"  GPU:          RTX 2050 (4 GB VRAM)")
    print(f"  Modelo:       yolov8{args.model} ({weights})")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch}")
    print(f"  Img size:     {args.img}")
    print(f"  Dataset:      {config.DATASET_YAML}")
    print("=" * 50)

    model = YOLO(weights)

    results = model.train(
        data=config.DATASET_YAML,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.img,
        project=config.MODEL_DIR,
        name="yolov8_train",
        exist_ok=True,
        device=0,
        workers=4,
        pretrained=True,
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        cos_lr=False,
        patience=100,
        save=True,
        save_period=-1,
        val=True,
        plots=True,
    )

    best_src = os.path.join(config.MODEL_DIR, "yolov8_train", "weights", "best.pt")
    if os.path.exists(best_src):
        shutil.copy(best_src, config.MODEL_PATH)
        print(f"\n[train] Modelo copiado para: {config.MODEL_PATH}")

    print("\n[train] Treinamento concluido!")
    print(f"[train] Metricas: {results.results_dict}")


if __name__ == "__main__":
    main()
