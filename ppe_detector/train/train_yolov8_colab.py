import os
import shutil
import yaml
import argparse

import kagglehub
from ultralytics import YOLO

CLASSES = [
    "no-safety-glove",
    "no-safety-helmet",
    "no-safety-shoes",
    "no-welding-glass",
    "safety-glove",
    "safety-helmet",
    "safety-shoes",
    "welding-glass",
]

MODEL_CHOICES = {
    "nano": "yolov8n.pt",
    "small": "yolov8s.pt",
    "medium": "yolov8m.pt",
    "large": "yolov8l.pt",
    "xlarge": "yolov8x.pt",
}

OUTPUT_DIR = "/content/ppe_detector_models"


def download_dataset():
    print("[train] Baixando dataset PPE do Kaggle...")
    path = kagglehub.dataset_download("anuragraj03/ppe-detection-m")
    print(f"[train] Dataset em: {path}")
    return path


def explore(dataset_path):
    print("[train] Explorando estrutura do dataset...")
    info = {"images_train": None, "labels_train": None,
            "images_val": None, "labels_val": None,
            "images_test": None, "labels_test": None}

    for root, dirs, files in os.walk(dataset_path):
        base = root.lower()
        if "train" in base:
            if "images" in base:
                info["images_train"] = root
            elif "labels" in base:
                info["labels_train"] = root
        elif "valid" in base:
            if "images" in base:
                info["images_val"] = root
            elif "labels" in base:
                info["labels_val"] = root
        elif "test" in base:
            if "images" in base:
                info["images_test"] = root
            elif "labels" in base:
                info["labels_test"] = root

    for split in ["train", "valid", "test"]:
        for kind in ["images", "labels"]:
            key = f"{kind}_{split}"
            if info.get(key) is None:
                candidate = os.path.join(dataset_path, split, kind)
                if os.path.isdir(candidate):
                    info[key] = candidate

    for k, v in info.items():
        print(f"       {k}: {v}")
    return info


def create_yaml(dataset_path, info):
    root = info["images_train"]
    if root:
        root = os.path.dirname(os.path.dirname(root))
    else:
        root = dataset_path

    data = {
        "path": root,
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 8,
        "names": CLASSES,
    }

    yaml_path = "/content/dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"[train] dataset.yaml criado em: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Treinar YOLOv8 no dataset de EPIs (Google Colab)")
    parser.add_argument("--model", type=str, default="small", choices=list(MODEL_CHOICES),
                        help="Tamanho do modelo YOLOv8")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Numero de epochs")
    parser.add_argument("--batch", type=int, default=16,
                        help="Tamanho do batch")
    parser.add_argument("--img", type=int, default=640,
                        help="Tamanho da imagem de entrada")
    args, _ = parser.parse_known_args()

    weights = MODEL_CHOICES[args.model]

    print("=" * 50)
    print("  PPE Detector - Treinamento YOLOv8 (Colab)")
    print("=" * 50)
    print(f"  GPU:          T4 / V100 / A100")
    print(f"  Modelo:       yolov8{args.model} ({weights})")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch}")
    print(f"  Img size:     {args.img}")
    print("=" * 50)

    dataset_path = download_dataset()
    info = explore(dataset_path)
    data_yaml = create_yaml(dataset_path, info)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = YOLO(weights)

    _drive_root = "/content/drive/MyDrive/ppe_detector_models"

    def backup_after_epoch(trainer):
        if not os.path.exists("/content/drive/MyDrive"):
            return
        os.makedirs(_drive_root, exist_ok=True)
        wdir = trainer.save_dir / "weights"
        for fname in ["best.pt", "last.pt"]:
            src = wdir / fname
            if src.exists():
                shutil.copy(str(src), os.path.join(_drive_root, fname))

    model.add_callback("on_train_epoch_end", backup_after_epoch)

    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.img,
        project=OUTPUT_DIR,
        name="yolov8_train",
        exist_ok=True,
        device=None,
        workers=2,
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

    best_src = os.path.join(OUTPUT_DIR, "yolov8_train", "weights", "best.pt")
    if os.path.exists(best_src):
        best_dst = os.path.join(OUTPUT_DIR, "best.pt")
        shutil.copy(best_src, best_dst)
        print(f"\n[train] Modelo salvo em: {best_dst}")
        print("[train] Baixe este arquivo para usar localmente.")
        print(f"[train] Execute no Colab: files.download('{best_dst}')")

    print("\n[train] Treinamento concluido!")
    print(f"[train] Metricas: {results.results_dict}")

    drive_backup = "/content/drive/MyDrive/ppe_detector_models"
    if os.path.exists("/content/drive/MyDrive"):
        os.makedirs(drive_backup, exist_ok=True)
        model_copy = os.path.join(OUTPUT_DIR, "best.pt")
        if os.path.exists(model_copy):
            shutil.copy(model_copy, os.path.join(drive_backup, "best.pt"))
        train_dir = os.path.join(OUTPUT_DIR, "yolov8_train")
        if os.path.exists(train_dir):
            shutil.copytree(train_dir, os.path.join(drive_backup, "yolov8_train"), dirs_exist_ok=True)
        print(f"[train] Backup salvo no Drive: {drive_backup}")


if __name__ == "__main__":
    main()
