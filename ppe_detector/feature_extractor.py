import os
import cv2
import numpy as np
import torch
from torchreid.utils import FeatureExtractor

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
OSNET_WEIGHTS = os.path.join(MODEL_DIR, "osnet_x0_25_msmt17.pth")
OSNET_URL = (
    "https://huggingface.co/kaiyangzhou/osnet/resolve/main/"
    "osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015"
    "_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
)

class Extractor:

    def __download_osnet(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        if os.path.exists(OSNET_WEIGHTS):
            return
        print("[OSNet] Baixando pesos (MSMT17) do HuggingFace...")
        from urllib.request import urlretrieve
        urlretrieve(OSNET_URL, OSNET_WEIGHTS)
        if not os.path.exists(OSNET_WEIGHTS):
            raise RuntimeError(
                "Falha ao baixar pesos OSNet do HuggingFace.\n"
                "Baixe manualmente de: https://huggingface.co/kaiyangzhou/osnet\n"
                f"e coloque em: {OSNET_WEIGHTS}"
            )
        print("[OSNet] Download concluido.")

    def __init__(self):
        self.__download_osnet()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        extractor = FeatureExtractor(
            model_name="osnet_x0_25",
            model_path=OSNET_WEIGHTS,
            device=device,
        )
        print(f"[OSNet] Carregado em {device}")
        self.__extractor = extractor

    def extract_embedding(self, crop_bgr: np.ndarray) -> np.ndarray:
        img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (128, 256))

        embedding = self.__extractor([img_rgb])[0].cpu().numpy().astype(np.float32)

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm

        return embedding