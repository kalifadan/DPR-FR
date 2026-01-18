from pathlib import Path

# ===========================
# General
# ===========================
SEED = 0
DEVICE = "cuda"
BATCH_SIZE = 64  # embedding batch size (not diffusion batch size)

# Face encoder wrapper config
FACE_BACKBONE_NAME = "facenet_inceptionresnetv1_vggface2"
RETURN_MODE = "embeddings"

BASE_DIR = Path(__file__).resolve().parents[1]  # project root

# ===========================
# LFW paths
# ===========================
LFW_IMAGES_ROOT = BASE_DIR / "data" / "lfw" / "lfw-deepfunneled" / "lfw-deepfunneled"
LFW_META_DIR = BASE_DIR / "data" / "lfw"

# Identity counts (required for ID protocol)
PEOPLE_CSV = LFW_META_DIR / "people.csv"

# Optional cache for embeddings
CACHE_DIR = BASE_DIR / "data" / "cache"
CACHE_TAG = "lfw_v18"

# Output directory
OUTPUT_DIR = BASE_DIR / "outputs" / "lfw_output_final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===========================
# Purifier (SDXL) settings
# ===========================
PURIFIER_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# Your knobs
PURIFIER_NUM_STEPS = 5         # TODO: 10
PURIFIER_DENOISING_START = 0.85     # (Diffusion strength is 1 - PURIFIER_DENOISING_START)
PURIFIER_NUM_VARIANTS = 1

# Practical settings
PURIFIER_RESOLUTION = 512
PURIFIER_BATCH_SIZE = 1

# ===========================
# Identification protocol
# ===========================
# Enrollment split:
# - include every identity with >= 2 images
# - choose 1 random probe image
# - enrollment images = all remaining images (optionally capped)
ENROLL_POLICY = "all_but_one"  # "all_but_one" (recommended) or "fixed_n"
ID_ENROLL_N = 3                # only used if ENROLL_POLICY="fixed_n"
MAX_ENROLL_PER_ID = None       # set e.g. 5 to cap compute; None = use all remaining

# Random resampling trials (average results)
ID_NUM_TRIALS = 1

# Optional speed caps (set None to use all)
MAX_KNOWN_IDENTITIES = None
MAX_UNKNOWN_IDENTITIES = None

# Similarity + threshold selection
SIMILARITY_METRIC = "cosine"   # dot product on L2-normalized embeddings
# Choose tau to maximize combined open-set accuracy on (known probes + unknown singletons)
THRESHOLD_POLICY = "max_open_set_acc"

# Unknown evaluation: identities with exactly 1 image (not enrolled)
EVAL_UNKNOWN_SINGLETONS = True

# ===========================
# Training/eval methods (three you requested)
# ===========================
# Each method specifies how to build enrollment templates and how to preprocess probes.
# - "baseline": clean enrollment only; clean probe
# - "clean_plus_diffusion": enrollment uses clean + 1 diffused per enrollment image (n+n); probe diffused
# - "diffusion": enrollment uses diffused-only per enrollment image; probe diffused
ID_METHODS = [
    dict(
        name="baseline",
        gallery_mode="clean_only",                 # clean_only
        gallery_diffused_variants_per_image=0,     # ignored for clean_only
        probe_mode="clean",                        # clean or diffused
    ),
    dict(
        name="baseline_probe_diffused",
        gallery_mode="clean_only",
        gallery_diffused_variants_per_image=0,
        probe_mode="diffused",
    ),
    dict(
        name="clean_plus_diffusion",
        gallery_mode="clean_plus_diffused",        # clean_plus_diffused => clean + K diffused per enroll image
        gallery_diffused_variants_per_image=1,     # K=1 gives n+n
        probe_mode="diffused",
    ),
    dict(
        name="diffusion",
        gallery_mode="diffused_only",              # diffuse each enroll image (K variants per image; usually 1)
        gallery_diffused_variants_per_image=1,
        probe_mode="diffused",
    ),
]

# ===========================
# Attacks (PGD) — run at test time in addition to clean evaluation
# ===========================
RUN_ATTACK_EVAL = True
ATTACK_EPS_LIST = [0.005, 0.01, 0.02]  # in [-1,1] space
ATTACK_ALPHA = 0.007
ATTACK_STEPS = 10

# Attack objectives:
# - known probes: "misidentify" (reduce true margin vs best other)
# - unknown singletons: "impersonate" (increase max_sim so it becomes accepted)
ATTACK_OBJECTIVE_KNOWN = "misidentify"
ATTACK_OBJECTIVE_UNKNOWN = "impersonate"
