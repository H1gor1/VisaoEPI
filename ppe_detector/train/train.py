"""
Treinamento do YOLOv5 com o dataset de EPIs.

Requisitos:
    - Conta no Kaggle (para baixar o dataset)
    - kagglehub instalado (pip install kagglehub)

Uso:
    python train.py                        # treino padrao (50 epochs)
    python train.py --epochs 30 --batch 8  # treino customizado
"""

import os
import sys
import shutil
import argparse
import subprocess

# permite importar config mesmo rodando de dentro de train/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def download_dataset():
    """Baixa o dataset do Kaggle usando kagglehub."""
    import kagglehub

    print("[train] Baixando dataset PPE do Kaggle...")
    path = kagglehub.dataset_download("anuragraj03/ppe-detection-m")
    print(f"[train] Dataset em: {path}")
    return path


def explore(dataset_path):
    """Explora a estrutura de pastas do dataset para achar images/ e labels/."""
    print("[train] Explorando estrutura do dataset...")

    info = {"images_train": None, "labels_train": None,
            "images_val": None, "labels_val": None,
            "yaml_file": None}

    for root, dirs, files in os.walk(dataset_path):
        for f in files:
            if f.endswith((".yaml", ".yml")):
                info["yaml_file"] = os.path.join(root, f)

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

    # fallback: estrutura plana comum
    for split in ["train", "valid", "test"]:
        for kind in ["images", "labels"]:
            key = f"{kind}_{split}" if split != "valid" else f"{kind}_val"
            if split == "valid":
                key = f"{kind}_val"
            else:
                key = f"{kind}_{split}"
            if info.get(key) is None:
                candidate = os.path.join(dataset_path, split, kind)
                if os.path.isdir(candidate):
                    info[key] = candidate

    for k, v in info.items():
        print(f"       {k}: {v}")
    return info


def create_yaml(dataset_path, info):
    """Gera dataset.yaml no padrao YOLOv5 com paths absolutos corrigidos."""
    # O yaml original do Kaggle tem paths do Windows (D:\...), entao
    #     # SEMPRE geramos um novo com os caminhos corretos desta maquina.
    os.makedirs(config.DATASET_DIR, exist_ok=True)

    # Determina a raiz real do dataset (pasta que contem train/ e valid/)
    root = info["images_train"]
    if root:
        root = os.path.dirname(os.path.dirname(root))  # sobe 2 niveis: images/ -> train/ -> raiz
    else:
        root = dataset_path

    data = {
        "path": root,
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 8,
        "names": config.CLASSES,
    }

    with open(config.DATASET_YAML, "w") as f:
        import yaml
        yaml.dump(data, f, default_flow_style=False)

    print(f"[train] dataset.yaml criado em: {config.DATASET_YAML}")
    print(f"       path: {root}")
    return config.DATASET_YAML


def clone_yolov5(base_dir):
    """Clona o repositorio oficial YOLOv5 se ainda nao existir."""
    repo_dir = os.path.join(base_dir, "yolov5_repo")

    if os.path.isdir(repo_dir):
        print(f"[train] YOLOv5 ja existe em: {repo_dir}")
        return repo_dir

    print("[train] Clonando YOLOv5...")
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/ultralytics/yolov5.git", repo_dir],
        check=True,
    )

    print("[train] Instalando dependencias do YOLOv5...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r",
         os.path.join(repo_dir, "requirements.txt")],
        check=True,
    )

    print(f"[train] YOLOv5 pronto em: {repo_dir}")
    return repo_dir


def train(yolov5_dir, data_yaml, epochs, batch, img_size, weights):
    """Executa o treinamento."""
    train_script = os.path.join(yolov5_dir, "train.py")

    cmd = [
        sys.executable, train_script,
        "--data", data_yaml,
        "--weights", weights,
        "--epochs", str(epochs),
        "--batch-size", str(batch),
        "--img", str(img_size),
        "--project", config.MODEL_DIR,
        "--name", "exp",
        "--exist-ok",
    ]

    print(f"[train] Comando: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def copy_best():
    """Copia best.pt do diretorio de experimento para models/best.pt."""
    src = os.path.join(config.MODEL_DIR, "exp", "weights", "best.pt")
    if os.path.exists(src):
        shutil.copy(src, config.MODEL_PATH)
        print(f"[train] Modelo copiado para: {config.MODEL_PATH}")
    else:
        print("[train] Atencao: best.pt nao encontrado no diretorio de saida.")


def main():
    parser = argparse.ArgumentParser(description="Treinar YOLOv5 no dataset de EPIs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--img", type=int, default=640)
    parser.add_argument("--weights", type=str, default="yolov5s.pt")
    args = parser.parse_args()

    print("=" * 50)
    print("  PPE Detector - Treinamento YOLOv5")
    print("=" * 50)

    # 1. baixa dataset
    dataset_path = download_dataset()

    # 2. explora estrutura
    info = explore(dataset_path)

    # 3. cria dataset.yaml
    data_yaml = create_yaml(dataset_path, info)

    # 4. clona YOLOv5
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yolov5_dir = clone_yolov5(base)

    # 5. treina
    print(f"\n[train] Iniciando treino: {args.epochs} epochs, batch {args.batch}, img {args.img}")
    train(yolov5_dir, data_yaml, args.epochs, args.batch, args.img, args.weights)

    # 6. copia modelo final
    copy_best()

    print("\n[train] Treinamento concluido com sucesso!")
    print(f"[train] Modelo salvo em: {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
