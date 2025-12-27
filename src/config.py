from pathlib import Path

SEED = 0
DEVICE = "cuda"
BATCH_SIZE = 32  # pairs mode embeds many images

# Face encoder wrapper config
FACE_BACKBONE_NAME = "facenet_inceptionresnetv1_vggface2"       # facenet_inceptionresnetv1_casia
RETURN_MODE = "embeddings"

BASE_DIR = Path(__file__).resolve().parents[1]  # project root

# LFW image root
LFW_IMAGES_ROOT = BASE_DIR / "data" / "lfw" / "lfw-deepfunneled" / "lfw-deepfunneled"

# LFW official protocol CSVs
LFW_META_DIR = BASE_DIR / "data" / "lfw"

LFW_MATCH_TRAIN = LFW_META_DIR / "matchpairsDevTrain.csv"
LFW_MISMATCH_TRAIN = LFW_META_DIR / "mismatchpairsDevTrain.csv"
LFW_MATCH_TEST = LFW_META_DIR / "matchpairsDevTest.csv"
LFW_MISMATCH_TEST = LFW_META_DIR / "mismatchpairsDevTest.csv"

# Optional cache for embeddings (recommended)
CACHE_DIR = BASE_DIR / "data" / "cache"
CACHE_TAG = "lfw_verif_v6"

# ---------------------------
# Purifier settings
# ---------------------------

PURIFIER_NAME = "none"
# PURIFIER_NAME = "sdxl"

# SDXL img2img model
PURIFIER_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
# PURIFIER_MODEL_ID = "stabilityai/sdxl-turbo"

# Your knobs
PURIFIER_NUM_STEPS = 4
PURIFIER_DENOISING_START = 0.6
PURIFIER_NUM_VARIANTS = 1

# Practical settings
PURIFIER_RESOLUTION = 512        # 512 is much faster than 1024
PURIFIER_BATCH_SIZE = 1          # keep 1 to avoid VRAM issues; raise if you have a lot of VRAM

# ---------------------------
# Attack settings
# ---------------------------

RUN_ATTACK_EVAL = True
ATTACK_MAX_TEST_PAIRS = None
ATTACK_EPS = 0.03
ATTACK_ALPHA = 0.007
ATTACK_STEPS = 10
PURIFY_BOTH_IN_VERIF = False

# MAX_TRAIN_PAIRS = 20
# MAX_TEST_PAIRS = 20
