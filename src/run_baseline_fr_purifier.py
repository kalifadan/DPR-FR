import hashlib
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import config
from face_recognition.fr_wrapper import FacialRecognitionWrapper
from purifiers import get_purifier


# ---------------------------
# LFW path helpers
# ---------------------------
def lfw_image_path(root: Path, name: str, imagenum: int) -> Path:
    # LFW filenames use 4-digit indexing: Name_0001.jpg
    fname = f"{name}_{int(imagenum):04d}.jpg"
    return root / name / fname


def _check_exists(paths: List[Path], max_missing_print: int = 10) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"[ERROR] Missing {len(missing)} image files. Examples:")
        for p in missing[:max_missing_print]:
            print("  -", p)
        raise FileNotFoundError("Some LFW image paths do not exist. Check LFW_IMAGES_ROOT and file naming.")


# ---------------------------
# Load official LFW protocol pairs
# ---------------------------
def load_pairs(match_csv: Path, mismatch_csv: Path) -> Tuple[List[Tuple[Path, Path]], torch.Tensor]:
    match_df = pd.read_csv(match_csv)
    mismatch_df = pd.read_csv(mismatch_csv)

    pairs: List[Tuple[Path, Path]] = []
    labels: List[int] = []

    # Match pairs: columns: name, imagenum1, imagenum2
    for _, r in match_df.iterrows():
        p1 = lfw_image_path(config.LFW_IMAGES_ROOT, r["name"], r["imagenum1"])
        p2 = lfw_image_path(config.LFW_IMAGES_ROOT, r["name"], r["imagenum2"])
        pairs.append((p1, p2))
        labels.append(1)

    # Mismatch pairs: columns: name, imagenum1, name.1, imagenum2
    for _, r in mismatch_df.iterrows():
        p1 = lfw_image_path(config.LFW_IMAGES_ROOT, r["name"], r["imagenum1"])
        p2 = lfw_image_path(config.LFW_IMAGES_ROOT, r["name.1"], r["imagenum2"])
        pairs.append((p1, p2))
        labels.append(0)

    y = torch.tensor(labels, dtype=torch.long)
    return pairs, y


# ---------------------------
# Embedding cache (image-level) — purifier-aware
# ---------------------------
def _safe_cache_ns(purifier_sig: str) -> str:
    if purifier_sig == "none":
        return "none"
    return hashlib.sha1(purifier_sig.encode("utf-8")).hexdigest()[:16]


def _cache_namespace(purifier_sig: str) -> str:
    return _safe_cache_ns(purifier_sig)


def _cache_key_for_image(path: Path, cache_ns: str) -> str:
    # Include purifier namespace + path + mtime
    s = f"{cache_ns}|{str(path)}|{path.stat().st_mtime_ns}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _cache_file_for_run(cache_ns: str) -> Path:
    # One cache file per backbone + purifier namespace + tag
    fname = f"emb_cache_{config.CACHE_TAG}_{config.FACE_BACKBONE_NAME}_{cache_ns}.pt"
    return config.CACHE_DIR / fname


def load_emb_cache(cache_ns: str) -> dict:
    p = _cache_file_for_run(cache_ns)
    if p.exists():
        payload = torch.load(p, map_location="cpu")
        if isinstance(payload, dict):
            print(f"[Cache] Loaded embedding cache: {p} (items={len(payload)})")
            return payload
    return {}


def save_emb_cache(cache: dict, cache_ns: str) -> None:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _cache_file_for_run(cache_ns)
    torch.save(cache, p)
    print(f"[Cache] Saved embedding cache: {p} (items={len(cache)})")


@torch.no_grad()
def embed_unique_images(fr: FacialRecognitionWrapper, image_paths: List[Path], purifier) -> dict:
    """
    Returns dict: cache_key -> embedding (CPU tensor, normalized).
    Applies purifier BEFORE embedding (even if purifier is NoOp).
    Uses persistent cache on disk to avoid recompute across runs.
    """
    fr = fr.to(config.DEVICE)
    cache_ns = _cache_namespace(purifier.signature)
    emb_cache = load_emb_cache(cache_ns)

    keys = [_cache_key_for_image(p, cache_ns) for p in image_paths]
    to_compute = [(p, k) for p, k in zip(image_paths, keys) if k not in emb_cache]

    if to_compute:
        print(f"[Embed] Computing {len(to_compute)} / {len(image_paths)} embeddings (rest from cache).")

    for i in tqdm(range(0, len(to_compute), config.BATCH_SIZE), desc="Embedding"):
        batch = to_compute[i : i + config.BATCH_SIZE]
        pil_imgs = [Image.open(p).convert("RGB") for (p, _) in batch]

        # Step 2: purification hook (NoOp now; SDXL later)
        pil_imgs = purifier(pil_imgs)

        out = fr(pil_imgs)  # (B,D)
        if not isinstance(out, torch.Tensor):
            out = out.astype("torch", batched=True)

        out = out.to(config.DEVICE)
        out = F.normalize(out, dim=-1).detach().cpu()

        for j, (_, key) in enumerate(batch):
            emb_cache[key] = out[j]

    if to_compute:
        save_emb_cache(emb_cache, cache_ns)

    return emb_cache


def pair_cosine_scores(pairs: List[Tuple[Path, Path]], emb_cache: dict, purifier_sig: str) -> torch.Tensor:
    cache_ns = _cache_namespace(purifier_sig)
    scores = []
    for p1, p2 in pairs:
        k1 = _cache_key_for_image(p1, cache_ns)
        k2 = _cache_key_for_image(p2, cache_ns)
        e1 = emb_cache[k1]
        e2 = emb_cache[k2]
        scores.append(torch.dot(e1, e2).item())
    return torch.tensor(scores, dtype=torch.float32)


# ---------------------------
# Threshold selection and evaluation
# ---------------------------
def pick_threshold_on_train(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Choose the threshold that maximizes accuracy on DevTrain.
    """
    uniq = torch.unique(scores).sort().values
    mids = (uniq[:-1] + uniq[1:]) / 2.0
    candidates = torch.cat([uniq[:1] - 1e-6, mids, uniq[-1:] + 1e-6])

    best_t = float(candidates[0].item())
    best_acc = -1.0

    for t in candidates:
        pred = (scores >= t).long()
        acc = (pred == labels).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_t = float(t.item())

    print(f"[Train] Best threshold on DevTrain: {best_t:.6f} | train-acc={best_acc:.4f}")
    return best_t


def evaluate(scores: torch.Tensor, labels: torch.Tensor, threshold: float, split_name: str) -> None:
    pred = (scores >= threshold).long()

    acc = (pred == labels).float().mean().item()

    tp = int(((pred == 1) & (labels == 1)).sum().item())
    tn = int(((pred == 0) & (labels == 0)).sum().item())
    fp = int(((pred == 1) & (labels == 0)).sum().item())
    fn = int(((pred == 0) & (labels == 1)).sum().item())

    print(f"[{split_name}] accuracy={acc:.4f}")
    print(f"[{split_name}] TP={tp} TN={tn} FP={fp} FN={fn}")


def main():
    # Validate required files exist
    for p in [
        config.LFW_IMAGES_ROOT,
        config.LFW_MATCH_TRAIN,
        config.LFW_MISMATCH_TRAIN,
        config.LFW_MATCH_TEST,
        config.LFW_MISMATCH_TEST,
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required path: {p}")

    # Official pairs
    train_pairs, y_train = load_pairs(config.LFW_MATCH_TRAIN, config.LFW_MISMATCH_TRAIN)
    test_pairs, y_test = load_pairs(config.LFW_MATCH_TEST, config.LFW_MISMATCH_TEST)

    # TODO: REMOVE BEFORE SUBMISSION, ONLY FOR QUICK SANITY
    train_pairs, y_train = train_pairs[:config.MAX_TRAIN_PAIRS], y_train[:config.MAX_TRAIN_PAIRS]
    test_pairs, y_test = test_pairs[:config.MAX_TEST_PAIRS], y_test[:config.MAX_TEST_PAIRS]

    print(f"Train pairs: {len(train_pairs)} | Test pairs: {len(test_pairs)}")

    # Unique images used by the protocol
    all_paths = sorted(set([p for pair in (train_pairs + test_pairs) for p in pair]))
    _check_exists(all_paths)

    # Face encoder
    fr = FacialRecognitionWrapper(
        backbone_name=config.FACE_BACKBONE_NAME,
        return_mode=config.RETURN_MODE,
    )

    # Step 1: purifier (NoOp now)
    purifier = get_purifier(
        name=getattr(config, "PURIFIER_NAME", "none"),
        num_steps=getattr(config, "PURIFIER_NUM_STEPS", 4),
        denoising_start=getattr(config, "PURIFIER_DENOISING_START", 0.6),
        num_variants=getattr(config, "PURIFIER_NUM_VARIANTS", 1),
    )
    print(f"[Purifier] Using: {purifier.signature}")

    # Embed all unique images (purifier-aware cache)
    emb_cache = embed_unique_images(fr, all_paths, purifier)

    # Score pairs
    train_scores = pair_cosine_scores(train_pairs, emb_cache, purifier.signature)
    test_scores = pair_cosine_scores(test_pairs, emb_cache, purifier.signature)

    # Train threshold only on DevTrain
    threshold = pick_threshold_on_train(train_scores, y_train)

    # Final results
    evaluate(train_scores, y_train, threshold, split_name="DevTrain")
    evaluate(test_scores, y_test, threshold, split_name="DevTest")


if __name__ == "__main__":
    torch.manual_seed(config.SEED)
    main()
