from pathlib import Path

ACTIVE_DATASET = "LFW_FOLDER"

SEED = 0
DEVICE = "cuda"
BATCH_SIZE = 16

FACE_BACKBONE_NAME = "facenet_inceptionresnetv1_vggface2"
RETURN_MODE = "embeddings"

KNN_K = 1
ACCEPT_THRESHOLD = 0.35

MIN_IMAGES_PER_ID = 2
ENROLL_PER_ID = 1
QUERY_PER_ID = 1

MAX_IDS = None  # e.g., 50 for quick tests

BASE_DIR = Path(__file__).resolve().parents[1]  # project root (since config.py is in src/)

DATASETS = {
    "LFW_FOLDER": {
        "type": "folder_identities",
        "root": BASE_DIR / "data" / "lfw" / "lfw-deepfunneled" / "lfw-deepfunneled",
    },
}

# Cache directory for enrollment templates (class map)
CACHE_DIR = BASE_DIR / "data" / "cache"
CACHE_TAG = "v1"  # bump if you change logic and want a fresh cache
