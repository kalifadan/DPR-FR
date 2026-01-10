from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any

import torch
import torch.nn.functional as F

from old import config


@dataclass
class ClassMap:
    """
    Enrollment cache:
      - templates: (num_ids, emb_dim) L2-normalized prototype embeddings
      - id_to_name: mapping numeric id -> identity name
      - meta: metadata to avoid accidental mismatches
    """
    templates: torch.Tensor
    id_to_name: Dict[int, str]
    meta: Dict[str, Any]


def make_cache_path(dataset_key: str, backbone_name: str) -> Path:
    """
    Build a cache filename that uniquely identifies the enrollment setting.
    (Later you can extend this with purifier settings, enrollment mode, etc.)
    """
    max_ids = "all" if config.MAX_IDS is None else str(config.MAX_IDS)
    fname = (
        f"class_map_{config.CACHE_TAG}_"
        f"{dataset_key}_"
        f"{backbone_name}_"
        f"seed{config.SEED}_"
        f"en{config.ENROLL_PER_ID}_"
        f"min{config.MIN_IMAGES_PER_ID}_"
        f"max{max_ids}.pt"
    )
    return config.CACHE_DIR / fname


def save_class_map(class_map: ClassMap, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "templates": class_map.templates.cpu(),
            "id_to_name": class_map.id_to_name,
            "meta": class_map.meta,
        },
        path,
    )


def load_class_map(path: Path) -> ClassMap:
    payload = torch.load(path, map_location="cpu")
    return ClassMap(
        templates=payload["templates"],
        id_to_name=payload["id_to_name"],
        meta=payload.get("meta", {}),
    )


def build_class_map(enroll_emb: torch.Tensor, enroll_labels: torch.Tensor, id_to_name: Dict[int, str]) -> ClassMap:
    """
    Build mean embedding per identity (prototype), then L2-normalize.
    enroll_emb: (N, D) on CPU
    enroll_labels: (N,) on CPU
    """
    if enroll_emb.ndim != 2:
        raise ValueError(f"Expected enroll_emb (N,D), got {enroll_emb.shape}")
    if enroll_labels.ndim != 1:
        raise ValueError(f"Expected enroll_labels (N,), got {enroll_labels.shape}")

    num_classes = int(enroll_labels.max().item()) + 1
    D = enroll_emb.shape[1]
    templates = torch.zeros((num_classes, D), dtype=enroll_emb.dtype)

    for c in range(num_classes):
        idx = (enroll_labels == c).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            raise RuntimeError(f"Class {c} has no enrollment samples (unexpected).")
        templates[c] = enroll_emb[idx].mean(dim=0)

    templates = F.normalize(templates, dim=-1)

    meta = {
        "dataset_key": config.ACTIVE_DATASET,
        "backbone": config.FACE_BACKBONE_NAME,
        "seed": config.SEED,
        "enroll_per_id": config.ENROLL_PER_ID,
        "min_images_per_id": config.MIN_IMAGES_PER_ID,
        "max_ids": config.MAX_IDS,
        "threshold": config.ACCEPT_THRESHOLD,
    }

    return ClassMap(templates=templates, id_to_name=id_to_name, meta=meta)


def build_or_load_class_map(
    dataset_key: str,
    backbone_name: str,
    enroll_emb: Optional[torch.Tensor] = None,
    enroll_labels: Optional[torch.Tensor] = None,
    id_to_name: Optional[Dict[int, str]] = None,
) -> ClassMap:
    """
    If cache exists -> load it.
    Else -> require enroll_emb/enroll_labels/id_to_name to build and save it.
    """
    cache_path = make_cache_path(dataset_key, backbone_name)
    if cache_path.exists():
        cm = load_class_map(cache_path)
        print(f"[ClassMap] Loaded cache: {cache_path}")
        return cm

    if enroll_emb is None or enroll_labels is None or id_to_name is None:
        raise ValueError("Cache missing, must provide enroll_emb, enroll_labels, and id_to_name to build it.")

    cm = build_class_map(enroll_emb=enroll_emb, enroll_labels=enroll_labels, id_to_name=id_to_name)
    save_class_map(cm, cache_path)
    print(f"[ClassMap] Built and saved cache: {cache_path}")
    return cm
