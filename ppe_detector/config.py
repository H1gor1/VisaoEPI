import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DATASET_YAML = os.path.join(DATASET_DIR, "dataset.yaml")
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "test_images")

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

VIOLATION_CLASSES = {0, 1, 2, 3}
SAFE_CLASSES = {4, 5, 6, 7}

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

COLOR_SAFE = (0, 255, 0)
COLOR_VIOLATION = (0, 0, 255)

CAMERA_ID = 0
VIDEO_OUTPUT_PATH = os.path.join(BASE_DIR, "output_video.avi")
VIDEO_FPS = 20
