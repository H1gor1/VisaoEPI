import os
import shutil

import kagglehub
import yaml

import config


def download_dataset():
    

    print("[train] Baixando dataset PPE do Kaggle...")
    path = kagglehub.dataset_download(
        "anuragraj03/ppe-detection-m",
        force_download=True,
        output_dir=config.DATASET_DIR
    )
    print(f"[train] Dataset em: {path}")
    return path

def create_yaml(dataset_path, info):
    os.makedirs(config.DATASET_DIR, exist_ok=True)

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
        "names": config.CLASSES,
    }

    with open(config.DATASET_YAML, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"[train] dataset.yaml criado em: {config.DATASET_YAML}")
    print(f"       path: {root}")
    return config.DATASET_YAML

def explore(dataset_path):
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

    for split in ["train", "valid", "test"]:
        for kind in ["images", "labels"]:
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

def copy_best():
    src = os.path.join(config.MODEL_DIR, "exp", "weights", "best.pt")
    if os.path.exists(src):
        shutil.copy(src, config.MODEL_PATH)
        print(f"[train] Modelo copiado para: {config.MODEL_PATH}")
    else:
        print("[train] Atencao: best.pt nao encontrado no diretorio de saida.")
