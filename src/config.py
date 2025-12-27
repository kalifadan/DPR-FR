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

# ---------------------------
# Purifier settings
# ---------------------------
# PURIFIER_NAME = "none"

PURIFIER_NAME = "sdxl"

# SDXL img2img model
PURIFIER_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# Your knobs
PURIFIER_NUM_STEPS = 4
PURIFIER_DENOISING_START = 0.6   # we will map this to img2img "strength" if needed
PURIFIER_NUM_VARIANTS = 1

# Practical settings
PURIFIER_RESOLUTION = 512        # 512 is much faster than 1024
PURIFIER_BATCH_SIZE = 1          # keep 1 to avoid VRAM issues; raise if you have a lot of VRAM
