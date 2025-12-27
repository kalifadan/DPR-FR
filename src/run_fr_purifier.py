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

import numpy as np


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    """
    x: (3,H,W) in [-1,1]
    returns PIL RGB
    """
    x = x.detach().cpu()
    x = (x + 1.0) / 2.0                # [0,1]
    x = torch.clamp(x, 0.0, 1.0)
    arr = (x.permute(1,2,0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def pgd_attack_pair(
    fr,
    x1: torch.Tensor,          # (1,3,H,W) in [-1,1]
    x2: torch.Tensor,          # (1,3,H,W) in [-1,1]
    y_same: int,               # 1 if same-person pair, 0 if mismatch
    eps: float = 0.03,         # in [-1,1] space (roughly 8/255 ≈ 0.0627)
    alpha: float = 0.007,      # step size
    steps: int = 10,
) -> torch.Tensor:
    """
    Returns adversarial x1_adv that tries to flip verification:
      - if y_same=1: reduce similarity (dodging)
      - if y_same=0: increase similarity (impersonation)
    """
    x1 = x1.to(fr.device_str)
    x2 = x2.to(fr.device_str)

    # fixed embedding for x2
    with torch.no_grad():
        e2 = fr.encode_tensor(x2)  # (1,D)

    # initialize
    x_adv = x1.clone().detach().requires_grad_(True)

    for _ in range(steps):
        e1 = fr.encode_tensor(x_adv)
        sim = (e1 * e2).sum(dim=-1)  # cosine since normalized

        # Objective:
        # same-pair: minimize sim
        # mismatch: maximize sim
        loss = sim.mean() if y_same == 0 else (-sim.mean())

        loss.backward()

        with torch.no_grad():
            grad = x_adv.grad
            x_adv = x_adv + alpha * torch.sign(grad)

            # Project to epsilon-ball around x1
            x_adv = torch.max(torch.min(x_adv, x1 + eps), x1 - eps)

            # Valid range [-1,1]
            x_adv = torch.clamp(x_adv, -1.0, 1.0)

        x_adv = x_adv.detach().requires_grad_(True)

    return x_adv.detach()


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

    # train_pairs, y_train = train_pairs[:config.MAX_TRAIN_PAIRS], y_train[:config.MAX_TRAIN_PAIRS]
    # test_pairs, y_test = test_pairs[:config.MAX_TEST_PAIRS], y_test[:config.MAX_TEST_PAIRS]

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

    if getattr(config, "RUN_ATTACK_EVAL", False):
        # optional subset
        if getattr(config, "ATTACK_MAX_TEST_PAIRS", None) is not None:
            n = int(config.ATTACK_MAX_TEST_PAIRS)
            test_pairs_sub = test_pairs[:n]
            y_test_sub = y_test[:n]
        else:
            test_pairs_sub = test_pairs
            y_test_sub = y_test

        # Build both modes
        purifier_none = get_purifier("none", num_steps=0, denoising_start=0.0)
        purifier_def = get_purifier(
            name=config.PURIFIER_NAME,
            num_steps=config.PURIFIER_NUM_STEPS,
            denoising_start=config.PURIFIER_DENOISING_START,
            num_variants=getattr(config, "PURIFIER_NUM_VARIANTS", 1),
        )

        # --- Calibrate thresholds separately on clean DevTrain ---
        emb_cache_base = embed_unique_images(fr, all_paths, purifier_none)
        train_scores_base = pair_cosine_scores(train_pairs, emb_cache_base, purifier_none.signature)
        thr_base = pick_threshold_on_train(train_scores_base, y_train)

        emb_cache_def = embed_unique_images(fr, all_paths, purifier_def)
        train_scores_def = pair_cosine_scores(train_pairs, emb_cache_def, purifier_def.signature)
        thr_def = pick_threshold_on_train(train_scores_def, y_train)

        print(f"[AttackEval] thr_base={thr_base:.6f} | thr_def={thr_def:.6f}")

        fooled_baseline = 0
        fooled_defended = 0
        correct_clean_base = 0
        correct_clean_def = 0
        total = len(test_pairs_sub)

        for (p1, p2), y in zip(test_pairs_sub, y_test_sub.tolist()):
            im1 = Image.open(p1).convert("RGB")
            im2 = Image.open(p2).convert("RGB")

            # tensors in [-1,1]
            x1 = fr.preprocess_pil([im1]).to(fr.device_str)
            x2 = fr.preprocess_pil([im2]).to(fr.device_str)

            # -------- clean (baseline) --------
            with torch.no_grad():
                e1 = fr.encode_tensor(x1)
                e2 = fr.encode_tensor(x2)
                sim_clean = float((e1 * e2).sum().item())
            pred_clean_base = int(sim_clean >= thr_base)
            correct_clean_base += int(pred_clean_base == y)

            # -------- clean (defended) --------
            if getattr(config, "PURIFY_BOTH_IN_VERIF", False):
                im1_def, im2_def = purifier_def([im1, im2])
            else:
                im1_def = purifier_def([im1])[0]
                im2_def = im2

            with torch.no_grad():
                e1d = fr([im1_def])
                e2d = fr([im2_def])
                sim_clean_def = float((e1d * e2d).sum().item())
            pred_clean_def = int(sim_clean_def >= thr_def)
            correct_clean_def += int(pred_clean_def == y)

            # -------- PGD attack on x1 --------
            x1_adv = pgd_attack_pair(
                fr=fr,
                x1=x1,
                x2=x2,
                y_same=y,
                eps=float(config.ATTACK_EPS),
                alpha=float(config.ATTACK_ALPHA),
                steps=int(config.ATTACK_STEPS),
            )

            # attacked baseline
            with torch.no_grad():
                e1a = fr.encode_tensor(x1_adv)
                sim_adv = float((e1a * e2).sum().item())
            pred_adv_base = int(sim_adv >= thr_base)
            fooled_baseline += int(pred_adv_base != y)

            # attacked defended
            adv_pil = tensor_to_pil(x1_adv[0])
            if getattr(config, "PURIFY_BOTH_IN_VERIF", False):
                adv_pil_def, im2_def2 = purifier_def([adv_pil, im2])
            else:
                adv_pil_def = purifier_def([adv_pil])[0]
                im2_def2 = im2

            with torch.no_grad():
                e1ad = fr([adv_pil_def])
                e2d2 = fr([im2_def2])
                sim_def = float((e1ad * e2d2).sum().item())
            pred_adv_def = int(sim_def >= thr_def)
            fooled_defended += int(pred_adv_def != y)

        # ASR (untargeted) == fooled rate
        asr_base = fooled_baseline / total
        asr_def = fooled_defended / total

        print(f"[AttackEval] Test pairs used: {total}")
        print(f"[AttackEval] Clean Acc (Baseline):  {correct_clean_base/total:.3f}")
        print(f"[AttackEval] Clean Acc (Defended):  {correct_clean_def/total:.3f}")
        print(f"[AttackEval] PGD ASR (Baseline):    {asr_base:.3f}  | Adv Acc: {(1-asr_base):.3f}")
        print(f"[AttackEval] PGD ASR (Defended):    {asr_def:.3f}  | Adv Acc: {(1-asr_def):.3f}")
        return


if __name__ == "__main__":
    torch.manual_seed(config.SEED)
    main()
