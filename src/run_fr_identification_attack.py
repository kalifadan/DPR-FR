import hashlib
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Use FR_CONFIG env var to point to the config module name (without .py) if desired
# Example: FR_CONFIG=config_id_final python run_fr_identification_attack.py
_cfg_name = os.environ.get("FR_CONFIG", "config_identification")
config = __import__(_cfg_name)

from face_recognition.fr_wrapper import FacialRecognitionWrapper
from purifiers import get_purifier


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lfw_image_path(root: Path, name: str, imagenum: int) -> Path:
    fname = f"{name}_{int(imagenum):04d}.jpg"
    return root / name / fname


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    """
    x: (3,H,W) in [-1,1]
    """
    x = x.detach().cpu()
    x = (x + 1.0) / 2.0
    x = torch.clamp(x, 0.0, 1.0)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------
# Embedding cache (image-level) — purifier-aware, variant-aware
# ---------------------------
def cache_path(cache_tag: str, backbone: str, cache_ns: str) -> Path:
    _ensure_dir(Path(config.CACHE_DIR))
    return Path(config.CACHE_DIR) / f"emb_cache_{cache_tag}_{backbone}_{cache_ns}.pt"


def load_cache(cache_fp: Path) -> Dict[str, torch.Tensor]:
    if cache_fp.exists():
        obj = torch.load(cache_fp, map_location="cpu")
        if isinstance(obj, dict):
            return obj
    return {}


def save_cache(cache_fp: Path, cache: Dict[str, torch.Tensor]) -> None:
    tmp = cache_fp.with_suffix(".tmp.pt")
    torch.save(cache, tmp)
    tmp.replace(cache_fp)


def cache_key(img_path: Path, variant_idx: int = 0) -> str:
    return f"{str(img_path)}::v{int(variant_idx)}"


@torch.no_grad()
def encode_pil(fr: FacialRecognitionWrapper, pil: Image.Image) -> torch.Tensor:
    """
    Returns (D,) L2-normalized embedding (CPU).
    """
    x = fr.preprocess_pil([pil]).to(fr.device_str)   # (1,3,H,W) in [-1,1]
    e = fr.encode_tensor(x)                          # (1,D)
    e = F.normalize(e, dim=1)
    return e[0].detach().cpu()


def embed_image(
    fr: FacialRecognitionWrapper,
    img_path: Path,
    purifier,
    cache: Dict[str, torch.Tensor],
    variant_idx: int = 0,
) -> torch.Tensor:
    """
    Embed a single image path. If purifier is not None, apply it to PIL first.
    Returns (D,) on CPU, L2-normalized.
    """
    k = cache_key(img_path, variant_idx=variant_idx)
    if k in cache:
        return cache[k]

    pil = Image.open(img_path).convert("RGB")
    if purifier is not None:
        pil = purifier([pil])[0]

    emb = encode_pil(fr, pil)
    cache[k] = emb
    return emb


# ---------------------------
# Split construction
# ---------------------------
def load_people_counts(people_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(people_csv)

    if "name" not in df.columns or "images" not in df.columns:
        raise ValueError(
            f"people.csv must contain columns ['name','images'], got {df.columns.tolist()}"
        )

    # Drop blank / malformed rows
    df = df.dropna(subset=["name", "images"]).copy()

    # Normalize
    df["name"] = df["name"].astype(str).str.strip()

    # Coerce to numeric then to int
    df["images"] = pd.to_numeric(df["images"], errors="coerce")
    df = df.dropna(subset=["images"]).copy()
    df["images"] = df["images"].astype(int)

    return df


def build_known_unknown_identities(df_people: pd.DataFrame) -> Tuple[List[str], List[str]]:
    known = df_people[df_people["images"] >= 2]["name"].tolist()
    unknown = df_people[df_people["images"] == 1]["name"].tolist()
    return known, unknown


def sample_enroll_probe(
    rng: np.random.Generator,
    num_images: int,
    enroll_policy: str,
    enroll_n: int,
    max_enroll: Optional[int],
) -> Tuple[List[int], int]:
    """
    Returns:
      enroll_nums: list of image numbers for enrollment
      probe_num: held-out image number
    """
    nums = np.arange(1, num_images + 1)
    probe_num = int(rng.choice(nums))
    remaining = [int(x) for x in nums if int(x) != probe_num]

    if enroll_policy == "all_but_one":
        enroll_nums = remaining
    elif enroll_policy == "fixed_n":
        if len(remaining) < enroll_n:
            return [], probe_num
        enroll_nums = [int(x) for x in rng.choice(remaining, size=enroll_n, replace=False)]
    else:
        raise ValueError(f"Unknown ENROLL_POLICY: {enroll_policy}")

    if max_enroll is not None and len(enroll_nums) > int(max_enroll):
        enroll_nums = [int(x) for x in rng.choice(enroll_nums, size=int(max_enroll), replace=False)]

    return enroll_nums, probe_num


# ---------------------------
# Threshold selection + AUC
# ---------------------------
def auc_roc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute ROC AUC without sklearn.
    """
    y_true = y_true.astype(np.int32)
    scores = scores.astype(np.float64)
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    all_scores = np.concatenate([pos, neg])
    order = np.argsort(all_scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(all_scores) + 1)

    # tie handling: average rank
    sorted_scores = all_scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = ranks[order[i:j + 1]].mean()
            ranks[order[i:j + 1]] = avg
        i = j + 1

    n_pos = len(pos)
    n_neg = len(neg)
    r_pos = ranks[:n_pos].sum()  # positives are first in concatenation
    auc = (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def pick_threshold_max_open_set_acc(
    known_max: np.ndarray,
    known_argmax: np.ndarray,
    known_true: np.ndarray,
    unk_max: Optional[np.ndarray],
) -> float:
    """
    Choose tau maximizing combined open-set accuracy:
      known correct if (max>=tau and argmax==true)
      unknown correct if (max<tau)
    """
    scores = known_max.copy()
    if unk_max is not None and len(unk_max) > 0:
        scores = np.concatenate([scores, unk_max])

    if len(scores) == 0:
        return 0.0
    if np.all(scores == scores[0]):
        return float(scores[0])

    qs = np.linspace(0.0, 1.0, 501)
    cands = np.unique(np.quantile(scores, qs))

    best_tau = float(cands[0])
    best_acc = -1.0

    for tau in cands:
        known_pred_known = known_max >= tau
        known_correct = np.sum(known_pred_known & (known_argmax == known_true))

        if unk_max is None:
            acc = known_correct / len(known_true)
        else:
            unk_correct = np.sum(unk_max < tau)
            acc = (known_correct + unk_correct) / (len(known_true) + len(unk_max))

        if acc > best_acc:
            best_acc = acc
            best_tau = float(tau)

    return best_tau


# ---------------------------
# PGD attacks (probe attacked pre-purifier; purifier applied after)
# ---------------------------
def pgd_attack_known_probe(
    fr: FacialRecognitionWrapper,
    x: torch.Tensor,                 # (1,3,H,W) in [-1,1]
    gallery: torch.Tensor,           # (N,D) on device, normalized
    true_idx: int,
    eps: float,
    alpha: float,
    steps: int,
) -> torch.Tensor:
    """
    Misidentify: minimize margin = sim_true - max_sim_other.
    """
    x0 = x.detach()
    adv = x.detach().clone().to(fr.device_str)

    for _ in range(steps):
        adv.requires_grad_(True)
        e = fr.encode_tensor(adv)
        e = F.normalize(e, dim=1)           # (1,D)
        sims = e @ gallery.t()              # (1,N)

        sim_true = sims[0, true_idx]
        sims_other = sims.clone()
        sims_other[0, true_idx] = -1e9
        sim_best_other = sims_other.max(dim=1).values[0]

        margin = sim_true - sim_best_other

        grad = torch.autograd.grad(margin, adv, retain_graph=False, create_graph=False)[0]
        adv = adv.detach() - alpha * grad.sign()

        adv = torch.max(torch.min(adv, x0 + eps), x0 - eps)
        adv = torch.clamp(adv, -1.0, 1.0)

    return adv.detach()


def pgd_attack_unknown_probe(
    fr: FacialRecognitionWrapper,
    x: torch.Tensor,                 # (1,3,H,W) in [-1,1]
    gallery: torch.Tensor,           # (N,D) on device, normalized
    eps: float,
    alpha: float,
    steps: int,
) -> torch.Tensor:
    """
    Impersonate: maximize max similarity to any gallery identity.
    """
    x0 = x.detach()
    adv = x.detach().clone().to(fr.device_str)

    for _ in range(steps):
        adv.requires_grad_(True)
        e = fr.encode_tensor(adv)
        e = F.normalize(e, dim=1)
        sims = e @ gallery.t()
        max_sim = sims.max(dim=1).values[0]

        loss = -max_sim
        grad = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]
        adv = adv.detach() - alpha * grad.sign()

        adv = torch.max(torch.min(adv, x0 + eps), x0 - eps)
        adv = torch.clamp(adv, -1.0, 1.0)

    return adv.detach()


def get_sdxl_purifier():
    return get_purifier(
        name="sdxl",
        num_steps=int(getattr(config, "PURIFIER_NUM_STEPS", 4)),
        denoising_start=float(getattr(config, "PURIFIER_DENOISING_START", 0.25)),
        num_variants=int(getattr(config, "PURIFIER_NUM_VARIANTS", 1)),
    )


def eval_one_trial(
    fr: FacialRecognitionWrapper,
    df_people: pd.DataFrame,
    method_cfg: dict,
    rng: np.random.Generator,
) -> dict:
    known_names_all, unknown_names_all = build_known_unknown_identities(df_people)

    # speed caps
    if getattr(config, "MAX_KNOWN_IDENTITIES", None) is not None:
        known_names = list(rng.choice(known_names_all, size=int(config.MAX_KNOWN_IDENTITIES), replace=False))
    else:
        known_names = known_names_all

    do_unknown = bool(getattr(config, "EVAL_UNKNOWN_SINGLETONS", True))
    if do_unknown:
        if getattr(config, "MAX_UNKNOWN_IDENTITIES", None) is not None:
            unknown_names = list(rng.choice(unknown_names_all, size=int(config.MAX_UNKNOWN_IDENTITIES), replace=False))
        else:
            unknown_names = unknown_names_all
    else:
        unknown_names = []

    counts = dict(zip(df_people["name"].tolist(), df_people["images"].tolist()))

    probe_needs_diff = (method_cfg["probe_mode"] == "diffused")
    gallery_mode = method_cfg["gallery_mode"]
    k_diff = int(method_cfg.get("gallery_diffused_variants_per_image", 0))

    # Purifier + caches
    purifier_sdxl = get_sdxl_purifier()

    cache_clean_fp = cache_path(config.CACHE_TAG, config.FACE_BACKBONE_NAME, "none")
    cache_clean = load_cache(cache_clean_fp)

    cache_ns_sdxl = _sha1(getattr(purifier_sdxl, "signature", "sdxl"))
    cache_sdxl_fp = cache_path(config.CACHE_TAG, config.FACE_BACKBONE_NAME, cache_ns_sdxl)
    cache_sdxl = load_cache(cache_sdxl_fp)

    # Build gallery templates (mean embeddings)
    gallery_names: List[str] = []
    gallery_embs: List[torch.Tensor] = []
    name_to_idx: Dict[str, int] = {}
    splits: Dict[str, Tuple[List[int], int]] = {}

    for name in tqdm(known_names, desc=f"[{method_cfg['name']}] Build gallery"):
        num_images = int(counts.get(name, 0))
        if num_images < 2:
            continue

        enroll_nums, probe_num = sample_enroll_probe(
            rng=rng,
            num_images=num_images,
            enroll_policy=str(config.ENROLL_POLICY),
            enroll_n=int(getattr(config, "ID_ENROLL_N", 3)),
            max_enroll=getattr(config, "MAX_ENROLL_PER_ID", None),
        )
        if len(enroll_nums) == 0:
            continue

        enroll_paths = [lfw_image_path(Path(config.LFW_IMAGES_ROOT), name, n) for n in enroll_nums]
        if not all(p.exists() for p in enroll_paths):
            continue

        # store split for later probe evaluation
        splits[name] = (enroll_nums, probe_num)

        embs = []

        if gallery_mode in ("clean_only", "clean_plus_diffused"):
            for p in enroll_paths:
                embs.append(embed_image(fr, p, purifier=None, cache=cache_clean, variant_idx=0))

        if gallery_mode in ("diffused_only", "clean_plus_diffused"):
            if gallery_mode == "diffused_only":
                kk = max(k_diff, 1)
            else:
                kk = max(k_diff, 0)
            for p in enroll_paths:
                for v in range(kk):
                    embs.append(embed_image(fr, p, purifier=purifier_sdxl, cache=cache_sdxl, variant_idx=v))

        E = torch.stack(embs, dim=0).mean(dim=0)
        E = F.normalize(E, dim=0)

        name_to_idx[name] = len(gallery_names)
        gallery_names.append(name)
        gallery_embs.append(E)

    if len(gallery_names) == 0:
        raise RuntimeError("Gallery is empty. Check LFW_IMAGES_ROOT and split policy.")

    gallery = torch.stack(gallery_embs, dim=0)  # (N,D) CPU

    # Build probes using stored splits (probe is the held-out image)
    known_true: List[int] = []
    known_probe_paths: List[Path] = []
    for name in gallery_names:
        if name not in splits:
            continue
        _, probe_num = splits[name]
        probe_path = lfw_image_path(Path(config.LFW_IMAGES_ROOT), name, probe_num)
        if probe_path.exists():
            known_true.append(name_to_idx[name])
            known_probe_paths.append(probe_path)

    # Clean evaluation on known probes
    known_max, known_argmax = [], []
    purifier_probe = purifier_sdxl if probe_needs_diff else None

    for p in tqdm(known_probe_paths, desc=f"[{method_cfg['name']}] Known probes (clean)"):
        pil = Image.open(p).convert("RGB")
        if purifier_probe is not None:
            pil = purifier_probe([pil])[0]
        e = encode_pil(fr, pil)  # (D,)
        sim_vec = (e[None, :] @ gallery.t()).squeeze(0)
        known_max.append(float(sim_vec.max().item()))
        known_argmax.append(int(sim_vec.argmax().item()))

    known_max = np.array(known_max, dtype=np.float64)
    known_argmax = np.array(known_argmax, dtype=np.int32)
    known_true_arr = np.array(known_true[: len(known_max)], dtype=np.int32)

    known_closed_acc = float(np.mean(known_argmax == known_true_arr))

    # Unknown singletons (clean)
    unk_max = None
    if do_unknown and len(unknown_names) > 0:
        scores = []
        for name in tqdm(unknown_names, desc=f"[{method_cfg['name']}] Unknown singletons (clean)"):
            p = lfw_image_path(Path(config.LFW_IMAGES_ROOT), name, 1)
            if not p.exists():
                continue
            pil = Image.open(p).convert("RGB")
            if purifier_probe is not None:
                pil = purifier_probe([pil])[0]
            e = encode_pil(fr, pil)
            sim_vec = (e[None, :] @ gallery.t()).squeeze(0)
            scores.append(float(sim_vec.max().item()))
        unk_max = np.array(scores, dtype=np.float64)

    # Select threshold
    tau = pick_threshold_max_open_set_acc(known_max, known_argmax, known_true_arr, unk_max if do_unknown else None)

    known_open_acc = float(np.mean((known_max >= tau) & (known_argmax == known_true_arr)))

    if do_unknown and unk_max is not None and len(unk_max) > 0:
        unknown_reject_acc = float(np.mean(unk_max < tau))
        y_auc = np.concatenate([np.ones_like(known_max, dtype=np.int32), np.zeros_like(unk_max, dtype=np.int32)])
        s_auc = np.concatenate([known_max, unk_max])
        auc_clean = auc_roc(y_auc, s_auc)
    else:
        unknown_reject_acc = float("nan")
        auc_clean = float("nan")

    # Attack evals
    attack_out = {}
    if getattr(config, "RUN_ATTACK_EVAL", False) and len(getattr(config, "ATTACK_EPS_LIST", [])) > 0:
        gallery_dev = F.normalize(gallery.to(fr.device_str), dim=1)

        # known probes PGD
        for eps in list(getattr(config, "ATTACK_EPS_LIST", [])):
            eps = float(eps)

            # Known probes: misidentify
            correct_k = 0
            known_scores_adv = []

            for p, t in tqdm(list(zip(known_probe_paths, known_true_arr.tolist())), desc=f"[{method_cfg['name']}] PGD known eps={eps}"):
                pil = Image.open(p).convert("RGB")
                x = fr.preprocess_pil([pil]).to(fr.device_str)

                x_adv = pgd_attack_known_probe(
                    fr=fr,
                    x=x,
                    gallery=gallery_dev,
                    true_idx=int(t),
                    eps=eps,
                    alpha=float(config.ATTACK_ALPHA),
                    steps=int(config.ATTACK_STEPS),
                )

                if purifier_probe is not None:
                    adv_pil = tensor_to_pil(x_adv[0])
                    adv_pil = purifier_probe([adv_pil])[0]
                    e = encode_pil(fr, adv_pil)
                else:
                    with torch.no_grad():
                        e = F.normalize(fr.encode_tensor(x_adv), dim=1)[0].detach().cpu()

                sim_vec = (e[None, :] @ gallery.t()).squeeze(0)
                max_sim = float(sim_vec.max().item())
                arg = int(sim_vec.argmax().item())
                known_scores_adv.append(max_sim)
                correct_k += int((max_sim >= tau) and (arg == int(t)))

            known_open_acc_adv = float(correct_k / max(len(known_true_arr), 1))

            # Unknown probes: impersonate
            unknown_reject_acc_adv = float("nan")
            auc_adv = float("nan")
            unk_scores_adv = None

            if do_unknown and len(unknown_names) > 0:
                correct_u = 0
                scores_u = []
                n_eval = 0
                for name in tqdm(unknown_names, desc=f"[{method_cfg['name']}] PGD unknown eps={eps}"):
                    p = lfw_image_path(Path(config.LFW_IMAGES_ROOT), name, 1)
                    if not p.exists():
                        continue
                    pil = Image.open(p).convert("RGB")
                    x = fr.preprocess_pil([pil]).to(fr.device_str)

                    x_adv = pgd_attack_unknown_probe(
                        fr=fr,
                        x=x,
                        gallery=gallery_dev,
                        eps=eps,
                        alpha=float(config.ATTACK_ALPHA),
                        steps=int(config.ATTACK_STEPS),
                    )

                    if purifier_probe is not None:
                        adv_pil = tensor_to_pil(x_adv[0])
                        adv_pil = purifier_probe([adv_pil])[0]
                        e = encode_pil(fr, adv_pil)
                    else:
                        with torch.no_grad():
                            e = F.normalize(fr.encode_tensor(x_adv), dim=1)[0].detach().cpu()

                    sim_vec = (e[None, :] @ gallery.t()).squeeze(0)
                    max_sim = float(sim_vec.max().item())
                    scores_u.append(max_sim)
                    correct_u += int(max_sim < tau)
                    n_eval += 1

                    if getattr(config, "MAX_UNKNOWN_IDENTITIES", None) is not None and n_eval >= int(config.MAX_UNKNOWN_IDENTITIES):
                        break

                unk_scores_adv = np.array(scores_u, dtype=np.float64)
                unknown_reject_acc_adv = float(correct_u / max(n_eval, 1))

                # AUC known vs unknown using attacked scores
                known_scores_adv_arr = np.array(known_scores_adv, dtype=np.float64)
                if len(unk_scores_adv) > 0:
                    y_auc = np.concatenate([np.ones_like(known_scores_adv_arr, dtype=np.int32), np.zeros_like(unk_scores_adv, dtype=np.int32)])
                    s_auc = np.concatenate([known_scores_adv_arr, unk_scores_adv])
                    auc_adv = auc_roc(y_auc, s_auc)

            attack_out[f"eps_{eps}"] = dict(
                known_open_acc=known_open_acc_adv,
                unknown_reject_acc=unknown_reject_acc_adv,
                auc_known_vs_unknown=auc_adv,
            )

    # Save caches
    save_cache(cache_clean_fp, cache_clean)
    save_cache(cache_sdxl_fp, cache_sdxl)

    return dict(
        method=method_cfg["name"],
        tau=float(tau),
        known_closed_acc=known_closed_acc,
        known_open_acc=known_open_acc,
        unknown_reject_acc=unknown_reject_acc,
        auc_known_vs_unknown=auc_clean,
        n_gallery=len(gallery_names),
        n_known=len(known_max),
        n_unknown=int(len(unk_max) if (unk_max is not None) else 0),
        attacks=attack_out,
    )


def main() -> None:
    set_seed(int(config.SEED))
    out_dir = Path(config.OUTPUT_DIR)
    _ensure_dir(out_dir)

    fr = FacialRecognitionWrapper(
        backbone_name=str(config.FACE_BACKBONE_NAME),
        return_mode=str(config.RETURN_MODE),
        device=str(config.DEVICE),
    )

    df_people = load_people_counts(Path(config.PEOPLE_CSV))

    rows = []
    for method in list(getattr(config, "ID_METHODS", [])):
        trial_outs = []
        for t in range(int(getattr(config, "ID_NUM_TRIALS", 1))):
            rng = np.random.default_rng(int(config.SEED) + 1000 * t + 17)
            trial_outs.append(eval_one_trial(fr, df_people, method, rng))

        def agg(key: str) -> Tuple[float, float]:
            xs = np.array([o[key] for o in trial_outs], dtype=np.float64)
            return float(np.nanmean(xs)), float(np.nanstd(xs))

        row = {"method": method["name"]}
        for k in ["tau", "known_closed_acc", "known_open_acc", "unknown_reject_acc", "auc_known_vs_unknown", "n_gallery", "n_known", "n_unknown"]:
            m, s = agg(k)
            row[f"{k}_mean"] = m
            row[f"{k}_std"] = s

        for eps in list(getattr(config, "ATTACK_EPS_LIST", [])):
            k = f"eps_{float(eps)}"
            for metric in ["known_open_acc", "unknown_reject_acc", "auc_known_vs_unknown"]:
                xs = np.array([o["attacks"].get(k, {}).get(metric, np.nan) for o in trial_outs], dtype=np.float64)
                row[f"{metric}_pgd_eps{eps}_mean"] = float(np.nanmean(xs))
                row[f"{metric}_pgd_eps{eps}_std"] = float(np.nanstd(xs))

        rows.append(row)

    df = pd.DataFrame(rows)

    # Table 1: known probes
    known_cols = ["method", "tau_mean", "tau_std", "known_closed_acc_mean", "known_closed_acc_std", "known_open_acc_mean", "known_open_acc_std"]
    for eps in list(getattr(config, "ATTACK_EPS_LIST", [])):
        known_cols += [f"known_open_acc_pgd_eps{eps}_mean", f"known_open_acc_pgd_eps{eps}_std"]
    table_known = df[known_cols].copy()

    # Table 2: unknown singletons
    unk_cols = ["method", "tau_mean", "tau_std", "unknown_reject_acc_mean", "unknown_reject_acc_std", "auc_known_vs_unknown_mean", "auc_known_vs_unknown_std"]
    for eps in list(getattr(config, "ATTACK_EPS_LIST", [])):
        unk_cols += [
            f"unknown_reject_acc_pgd_eps{eps}_mean", f"unknown_reject_acc_pgd_eps{eps}_std",
            f"auc_known_vs_unknown_pgd_eps{eps}_mean", f"auc_known_vs_unknown_pgd_eps{eps}_std",
        ]
    table_unk = df[unk_cols].copy()

    fp_known = out_dir / "table_known_probes.csv"
    fp_unk = out_dir / "table_unknown_singletons.csv"
    table_known.to_csv(fp_known, index=False)
    table_unk.to_csv(fp_unk, index=False)

    print("\n=== Known probes ===")
    print(table_known.to_string(index=False))
    print("\n=== Unknown singletons ===")
    print(table_unk.to_string(index=False))
    print(f"\nSaved:\n- {fp_known}\n- {fp_unk}")


if __name__ == "__main__":
    main()
