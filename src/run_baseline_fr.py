import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

import config

from face_recognition.fr_wrapper import FacialRecognitionWrapper
from class_map_cache import build_or_load_class_map, make_cache_path


# ---------------------------
# Dataset loading utilities
# ---------------------------
def _list_identity_dirs(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])


def _list_images(person_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in person_dir.rglob("*") if p.suffix.lower() in exts])


def _load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_folder_identities_dataset(
    root: Path,
) -> Tuple[List[Image.Image], List[int], List[Image.Image], List[int], Dict[int, str]]:
    """
    For each identity folder: pick ENROLL_PER_ID images for enrollment and QUERY_PER_ID for query.
    Minimal baseline: 1 enroll image + 1 query image per identity (configurable).
    """
    rng = random.Random(config.SEED)

    enroll_imgs: List[Image.Image] = []
    enroll_labels: List[int] = []
    query_imgs: List[Image.Image] = []
    query_labels: List[int] = []
    id_to_name: Dict[int, str] = {}

    class_id = 0
    for person_dir in _list_identity_dirs(root):
        imgs = _list_images(person_dir)
        if len(imgs) < config.MIN_IMAGES_PER_ID:
            continue

        rng.shuffle(imgs)

        need = config.ENROLL_PER_ID + config.QUERY_PER_ID
        if len(imgs) < need:
            continue

        enroll_paths = imgs[: config.ENROLL_PER_ID]
        query_paths = imgs[config.ENROLL_PER_ID : config.ENROLL_PER_ID + config.QUERY_PER_ID]

        for p in enroll_paths:
            enroll_imgs.append(_load_rgb(p))
            enroll_labels.append(class_id)

        for p in query_paths:
            query_imgs.append(_load_rgb(p))
            query_labels.append(class_id)

        id_to_name[class_id] = person_dir.name
        class_id += 1

        if config.MAX_IDS is not None and class_id >= config.MAX_IDS:
            break

    if class_id < 2:
        raise RuntimeError("Need at least 2 identities with enough images.")

    return enroll_imgs, enroll_labels, query_imgs, query_labels, id_to_name


# ---------------------------
# Embedding + KNN baseline
# ---------------------------
@torch.no_grad()
def embed_images(fr: FacialRecognitionWrapper, images: List[Image.Image]) -> torch.Tensor:
    fr = fr.to(config.DEVICE)
    all_embs: List[torch.Tensor] = []

    for i in range(0, len(images), config.BATCH_SIZE):
        batch = images[i : i + config.BATCH_SIZE]

        out = fr(batch)  # expected (B, D) or compatible object
        if isinstance(out, torch.Tensor):
            embs = out
        else:
            # If your wrapper uses a different return type, adjust here
            embs = out.astype("torch", batched=True)

        embs = embs.to(config.DEVICE)
        embs = F.normalize(embs, dim=-1)
        all_embs.append(embs.detach().cpu())

    return torch.cat(all_embs, dim=0)


def knn_predict(query_emb: torch.Tensor, templates: torch.Tensor) -> Tuple[List[int], List[float]]:
    # cosine similarity since embeddings are normalized
    sims = query_emb @ templates.t()  # (N, C)
    scores, preds = sims.max(dim=1)
    return preds.tolist(), scores.tolist()


def main():
    ds_cfg = config.DATASETS[config.ACTIVE_DATASET]
    ds_type = ds_cfg["type"]

    if ds_type == "folder_identities":
        enroll_imgs, enroll_lbls, query_imgs, query_lbls, id_to_name = load_folder_identities_dataset(ds_cfg["root"])
    else:
        raise ValueError(f"Unknown dataset type: {ds_type}")

    print(f"Dataset: {config.ACTIVE_DATASET}")
    print(f"Identities: {len(id_to_name)} | Enroll images: {len(enroll_imgs)} | Query images: {len(query_imgs)}")

    fr = FacialRecognitionWrapper(
        backbone_name=config.FACE_BACKBONE_NAME,
        return_mode=config.RETURN_MODE,
    )

    # ---------------------------
    # PART 2: Class map cache (enrollment templates)
    # ---------------------------
    cache_path = make_cache_path(config.ACTIVE_DATASET, config.FACE_BACKBONE_NAME)

    if cache_path.exists():
        class_map = build_or_load_class_map(
            dataset_key=config.ACTIVE_DATASET,
            backbone_name=config.FACE_BACKBONE_NAME,
        )
        templates = class_map.templates
        id_to_name = class_map.id_to_name  # load mapping from cache to avoid mismatch
    else:
        enroll_emb = embed_images(fr, enroll_imgs)
        enroll_labels_t = torch.tensor(enroll_lbls, dtype=torch.long)

        class_map = build_or_load_class_map(
            dataset_key=config.ACTIVE_DATASET,
            backbone_name=config.FACE_BACKBONE_NAME,
            enroll_emb=enroll_emb,
            enroll_labels=enroll_labels_t,
            id_to_name=id_to_name,
        )
        templates = class_map.templates

    # Query embeddings (always computed; later we will cache per-attack if needed)
    query_emb = embed_images(fr, query_imgs)

    preds, scores = knn_predict(query_emb, templates)

    # Metrics (simple baseline)
    total = len(query_lbls)
    correct = sum(int(p == gt) for p, gt in zip(preds, query_lbls))
    accepted = sum(int(sc >= config.ACCEPT_THRESHOLD) for sc in scores)
    accepted_correct = sum(
        int((sc >= config.ACCEPT_THRESHOLD) and (p == gt))
        for p, gt, sc in zip(preds, query_lbls, scores)
    )

    print(f"Top-1 accuracy: {correct/total:.3f}")
    print(f"Accepted fraction (@thr={config.ACCEPT_THRESHOLD}): {accepted/total:.3f}")
    if accepted > 0:
        print(f"Wrong-among-accepted (proxy): {(accepted - accepted_correct)/accepted:.3f}")
    else:
        print("Wrong-among-accepted (proxy): N/A (no accepted)")

    # Show a few examples
    for i in range(min(10, total)):
        gt_name = id_to_name[query_lbls[i]]
        pred_name = id_to_name[preds[i]]
        print(f"[{i}] GT={gt_name:22s} Pred={pred_name:22s} sim={scores[i]:.3f}")


if __name__ == "__main__":
    torch.manual_seed(config.SEED)
    main()
