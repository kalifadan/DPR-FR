from pathlib import Path

SEED = 0
DEVICE = "cuda"
BATCH_SIZE = 32  # pairs mode embeds many images; 32 is usually fine

# Face encoder wrapper config (matches the wrapper we created)
FACE_BACKBONE_NAME = "facenet_inceptionresnetv1_vggface2"
RETURN_MODE = "embeddings"

BASE_DIR = Path(__file__).resolve().parents[1]  # project root

# LFW image root (folder containing person subfolders)
LFW_IMAGES_ROOT = BASE_DIR / "data" / "lfw" / "lfw-deepfunneled" / "lfw-deepfunneled"

# LFW official protocol CSVs (put them in data/lfw/metadata/ for example)
LFW_META_DIR = BASE_DIR / "data" / "lfw"

LFW_MATCH_TRAIN = LFW_META_DIR / "matchpairsDevTrain.csv"
LFW_MISMATCH_TRAIN = LFW_META_DIR / "mismatchpairsDevTrain.csv"
LFW_MATCH_TEST = LFW_META_DIR / "matchpairsDevTest.csv"
LFW_MISMATCH_TEST = LFW_META_DIR / "mismatchpairsDevTest.csv"

# Optional cache for embeddings (recommended)
CACHE_DIR = BASE_DIR / "data" / "cache"
CACHE_TAG = "lfw_verif_v1"
