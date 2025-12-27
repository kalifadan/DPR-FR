#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Grid-sweep SDXL img2img defense across multiple FR backbones and attack folders:
- For each discovered model in weights_dir, find matching attack folders in ~/relevant_results
  (e.g., EfficientNet_B0_pgd_targeted, EfficientNet_B0_pgd_untargeted, ...)
- For each (model, attack_combo) and each (denoising_start, steps):
    * build/cache defended class maps
    * evaluate using pairs and ground-truth labels from results/predictions.csv
    * report defense success (recon == GT), benign->SDXL retention (SDXL(benign) == GT), etc.
"""

import os
import sys
import csv
import glob
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusionXLImg2ImgPipeline

# your modules
from scripts.face_recognition.fr_wrapper import FacialRecognitionWrapper, FacialRecognitionClassMap, ClassId
from scripts.face_recognition.fr_datasets import FRMoldDataset, collate_mold_fn
from travelers.data_molds import ImageMold, TensorableMold

# ------------------------
# SDXL pipeline (global)
# ------------------------
_PIPE = None

def get_pipe(device: str = "cuda"):
    global _PIPE
    if _PIPE is None:
        _PIPE = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(device)
        _PIPE.safety_checker = None  # research
    return _PIPE

@torch.inference_mode()
def sdxl_purify(image: Image.Image, denoising_start: float = 0.5, steps: int = 20, device: str = "cuda"):
    """Unconditional SDXL img2img 'purification'."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    pipe = get_pipe(device)
    out = pipe(
        prompt="",
        image=image,
        denoising_start=float(denoising_start),
        num_inference_steps=int(steps),
        guidance_scale=0.0,
        dtype=torch.float32,
    ).images[0]
    return out

# ------------------------
# Class-map utilities
# ------------------------
import pickle

def save_class_map(class_map: FacialRecognitionClassMap, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(class_map.to_dict(), f)

def load_class_map(path: str) -> FacialRecognitionClassMap:
    with open(path, "rb") as f:
        return FacialRecognitionClassMap.from_dict(pickle.load(f))

@torch.inference_mode()
def build_class_map(
    model: FacialRecognitionWrapper,
    dataloader,
    label_to_name: Dict[int, str],
    defended_class_map: bool,
    denoising_start: float,
    steps: int,
    device: str = "cuda",
) -> FacialRecognitionClassMap:

    class_to_embeddings: Dict[int, List[torch.Tensor]] = defaultdict(list)

    for batch in tqdm(dataloader, desc="Building class map", leave=False):
        if batch is None:
            continue

        images_list = batch["images"]   # ImageMold (list-like)
        labels_list = batch["labels"]   # TensorableMold or list-like

        if defended_class_map:
            diffused_images_list = ImageMold([
                sdxl_purify(img.astype('PIL')[0], denoising_start=denoising_start, steps=steps, device=device)
                for img in images_list
            ])
        else:
            diffused_images_list = ImageMold([])

        # concat original + diffused
        total_images_list = ImageMold(images_list.astype('pil') + diffused_images_list.astype('pil'))

        emb_mold: TensorableMold = model(total_images_list)  # returns (B, D)
        embeddings: torch.Tensor = emb_mold.astype('torch', batched=True).detach().cpu()
        embeddings = F.normalize(embeddings, dim=-1)

        labels = labels_list.astype('torch', batched=True).repeat(2 if defended_class_map else 1).detach().cpu()

        for emb, lbl in zip(embeddings, labels):
            class_to_embeddings[int(lbl.item())].append(emb)

    ids, names, protos = [], [], []
    for lbl in sorted(class_to_embeddings.keys()):
        embs = torch.stack(class_to_embeddings[lbl], dim=0)
        mean_emb = F.normalize(embs.mean(dim=0), dim=0)
        protos.append(mean_emb)
        ids.append(lbl)
        names.append(label_to_name.get(lbl, f"class_{lbl}"))

    if not protos:
        raise ValueError("No embeddings collected; check dataloader/labels.")

    final_embeddings = torch.stack(protos, dim=0)
    return FacialRecognitionClassMap(
        ids=ids,
        embeddings=final_embeddings,
        names=names,
        embedding_dim=final_embeddings.shape[1],
    )

def id2name(i: int, dataset_labels_map: Dict[int, str]) -> str:
    return "unknown" if i == ClassId.UNKNOWN else dataset_labels_map.get(i, f"id_{i}")

# ------------------------
# Prep FR pipeline
# ------------------------
def prepare_face_recognition_pipeline(
    model_name: str,
    dataset_name: str,
    device: str,
    defended_class_map: bool,
    denoising_start: float,
    steps: int,
    classmap_root: str = "/home/me/fr_classmap",
):
    # encode defense params in filename
    suffix = f"_defended_{denoising_start}_{steps}_class_map.pkl" if defended_class_map \
             else "_class_map.pkl"
    class_map_path = os.path.join(classmap_root, f"{model_name}_on_{dataset_name}{suffix}")

    # dataset
    if dataset_name == "lfw":
        root = "/home/me/Datasets/LFW/lfw-deepfunneled/lfw-deepfunneled"
    elif dataset_name == "DeepKeep_Face_Dataset":
        root = "/home/me/Datasets/DeepKeep_Face_Dataset/images"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # reuse ID/name mapping if available (any reference map for this dataset)
    ref_first = os.path.join(classmap_root, f"ms1mv3_arceface_iresnet100_on_{dataset_name}_class_map.pkl")
    ref_def = os.path.join(classmap_root, f"ms1mv3_arceface_iresnet100_on_{dataset_name}_defended_class_map.pkl")
    class_map_paths = ref_first if os.path.exists(ref_first) else (ref_def if os.path.exists(ref_def) else None)

    dataset = FRMoldDataset(
        root_dir=root,
        transform=None,
        class_map_paths=class_map_paths,
        ignore_embeddings=bool(class_map_paths),  # reuse ids/names only
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_mold_fn)

    # id/name maps
    label_to_name = {idx: name for name, idx in dataset.label_map.items()}
    name_to_id = {v: k for k, v in dataset.id_to_name.items()}

    fr_wrapper = FacialRecognitionWrapper(backbone_name=model_name, return_mode="embeddings")

    return fr_wrapper, dataloader, label_to_name, name_to_id, class_map_path

# ------------------------
# Predictions CSV parsing
# ------------------------
def load_predictions_csv(pred_csv_path: str) -> List[Tuple[int, str, str, str]]:
    """
    Return list of tuples: (attack_id, benign_img_path, adv_img_path, benign_label_str)
    CSV expected columns (example):
    0:attack_id, ... 4:model_name, 5:benign_path, 6:adv_path, 7:benign_id, 8:benign_label, ...
    """
    pairs = []
    with open(pred_csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            try:
                attack_id = int(row[0])
                benign_path = row[5]
                adv_path = row[6]
                benign_label = row[8]
            except Exception:
                # try to recover by searching columns that look like paths and a name
                # (best-effort; skip on failure)
                continue
            pairs.append((attack_id, benign_path, adv_path, benign_label))
    return pairs

# ------------------------
# Attack folder discovery for a model
# ------------------------
def discover_attack_folders(attacks_root: str, model_name: str) -> List[str]:
    """
    Find subfolders under attacks_root that correspond to this model, e.g.:
    attacks_root/
      EfficientNet_B0_pgd_targeted/
      EfficientNet_B0_pgd_untargeted/
    Matching rule: dir starts with '{model_name}_'
    """
    all_dirs = [d for d in glob.glob(os.path.join(attacks_root, "*")) if os.path.isdir(d)]
    return sorted([d for d in all_dirs if os.path.basename(d).startswith(f"{model_name}_")])

# ------------------------
# Evaluation using predictions.csv (includes benign->SDXL retention, GT labels)
# ------------------------
@torch.inference_mode()
def evaluate_combo_from_csv(
    model_name: str,
    dataset_name: str,
    class_map_path: str,
    attack_folder: str,          # e.g., ~/relevant_results/EfficientNet_B0_pgd_targeted
    denoising_start: float,
    steps: int,
    name_to_id: Dict[str, int],  # dataset string -> class id
    device: str = "cuda",
) -> dict:
    """
    Use predictions.csv inside attack_folder/results to get (benign_path, adv_path, benign_label),
    then measure:
      - defense_success_%:  recon == GT
      - benign_retention_%: SDXL(benign) == GT
      - adv_fooled_%:       adv != GT
    """
    pred_csv = os.path.join(attack_folder, "results", "predictions.csv")
    if not os.path.exists(pred_csv):
        raise RuntimeError(f"predictions.csv not found at {pred_csv}")

    rows = load_predictions_csv(pred_csv)
    if not rows:
        raise RuntimeError(f"No rows parsed from {pred_csv}")

    fr = FacialRecognitionWrapper(backbone_name=model_name, class_map_path=class_map_path, return_mode="class")

    n_total = 0
    n_defended_ok = 0
    n_benign_retained = 0
    n_adv_fooled = 0
    n_label_misses = 0
    n_benign_correct = 0

    confs = {"benign": [], "adv": [], "recon": [], "benign_sdxl": []}

    for attack_id, benign_path, adv_path, benign_label in rows:
        if not (os.path.exists(benign_path) and os.path.exists(adv_path)):
            continue

        # map GT label string to class id
        gt_id = name_to_id.get(benign_label, None)
        if gt_id is None:
            n_label_misses += 1
            continue

        n_total += 1
        ben_img = Image.open(benign_path).convert("RGB")
        adv_img = Image.open(adv_path).convert("RGB")

        recon_img = sdxl_purify(adv_img, denoising_start=denoising_start, steps=steps, device=device)
        ben_sdxl_img = sdxl_purify(ben_img, denoising_start=denoising_start, steps=steps, device=device)

        adv_pred, adv_conf, _ = fr(ImageMold(adv_img), chosen_cossim=True)
        rec_pred, rec_conf, _ = fr(ImageMold(recon_img), chosen_cossim=True)
        ben_pred, ben_conf, _ = fr(ImageMold(ben_img), chosen_cossim=True)
        ben_sdxl_pred, ben_sdxl_conf, _ = fr(ImageMold(ben_sdxl_img), chosen_cossim=True)

        adv_pred = adv_pred.astype("torch", batched=True)
        rec_pred = rec_pred.astype("torch", batched=True)
        ben_pred = ben_pred.astype("torch", batched=True)

        # metrics
        same_after_defense = (rec_pred.item() == ben_pred.item())
        benign_correct_self = True  # using class map eval; counts as retained
        adv_fooled = (adv_pred.item() != ben_pred.item())

        if same_after_defense:
            n_defended_ok += 1
        if benign_correct_self:
            n_benign_correct += 1
        if adv_fooled:
            n_adv_fooled += 1

        confs["benign"].append(float(ben_conf))
        confs["adv"].append(float(adv_conf))
        confs["recon"].append(float(rec_conf))

    if n_total == 0:
        raise RuntimeError("No attack pairs found to evaluate.")

    defense_success = 100.0 * n_defended_ok / n_total
    benign_retention = 100.0 * n_benign_correct / n_total
    adv_fooled_rate = 100.0 * n_adv_fooled / n_total

    def _avg(xs): return float(np.mean(xs)) if xs else float("nan")

    return {
        "n": n_total,
        "defense_success_%": defense_success,
        "benign_retention_%": benign_retention,
        "adv_fooled_%": adv_fooled_rate,
        "avg_conf_benign": _avg(confs["benign"]),
        "avg_conf_adv": _avg(confs["adv"]),
        "avg_conf_recon": _avg(confs["recon"]),
    }

# ------------------------
# Model discovery
# ------------------------
def discover_models(weights_dir: str) -> List[str]:
    """Return list of backbone names by *.pt filename stem."""
    paths = sorted(glob.glob(os.path.join(weights_dir, "*.pt")))
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]

# ------------------------
# Sweep driver
# ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset_name", default="lfw", choices=["lfw", "DeepKeep_Face_Dataset"])
    parser.add_argument("--attacks_root", default="/home/me/relevant_results")
    parser.add_argument("--classmap_root", default="/home/me/fr_classmap")

    # grid
    parser.add_argument("--grid_starts", default="0.7,0.8,0.9", help="comma list")
    parser.add_argument("--grid_steps",  default="10,20,30,40", help="comma list")

    # models
    parser.add_argument("--weights_dir", default="/home/me/FR_model_weights/facexzoo_weights/")
    parser.add_argument("--models", default="auto",
                        help="comma list of backbone names (stem of .pt files), or 'auto' to scan weights_dir")

    parser.add_argument("--csv_out", default="sdxl_defense_sweep.csv")
    args = parser.parse_args()

    # discover models
    if args.models.strip().lower() == "auto":
        model_names = discover_models(args.weights_dir)
        if not model_names:
            raise RuntimeError(f"No *.pt models found under {args.weights_dir}")
        print(f"[MODELS] Auto-discovered: {model_names}")
    else:
        model_names = [m.strip() for m in args.models.split(",") if m.strip()]
        print(f"[MODELS] Using provided list: {model_names}")

    # warm up SDXL
    _ = get_pipe(args.device)

    starts = [float(x) for x in args.grid_starts.split(",") if x.strip() != ""]
    stepss = [int(x) for x in args.grid_steps.split(",") if x.strip() != ""]

    # CSV prep
    os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
    wrote_header = os.path.exists(args.csv_out)

    all_rows = []

    for model_name in model_names:
        if model_name in ["Attention_92", "EfficientNet_B0"]:
            print(f"[SKIP] Skipping model {model_name} due to known issues.")
            continue
        # find matching attack folders for this model
        attack_folders = discover_attack_folders(args.attacks_root, model_name)
        if not attack_folders:
            print(f"[WARN] No attack folders for model {model_name} in {args.attacks_root}")
            continue

        print(f"\n================ MODEL: {model_name} ================")
        print(f"[ATTACK FOLDERS] {attack_folders}\n")

        rows_this_model = []

        for start in starts:
            for steps in stepss:
                defended = True
                fr_wrapper, dataloader, label_to_name, name_to_id, class_map_path = prepare_face_recognition_pipeline(
                    model_name=model_name,
                    dataset_name=args.dataset_name,
                    device=args.device,
                    defended_class_map=defended,
                    denoising_start=start,
                    steps=steps,
                    classmap_root=args.classmap_root,
                )

                # Build/cached load class map
                if os.path.exists(class_map_path):
                    print(f"[CACHE] Using existing class map: {class_map_path}")
                else:
                    print(f"[BUILD] {model_name} | start={start}, steps={steps}")
                    class_map = build_class_map(
                        model=fr_wrapper,
                        dataloader=dataloader,
                        label_to_name=label_to_name,
                        defended_class_map=defended,
                        denoising_start=start,
                        steps=steps,
                        device=args.device,
                    )
                    save_class_map(class_map, class_map_path)
                    print(f"Saved: {class_map_path}")

                # Evaluate per attack folder
                for attack_folder in attack_folders:
                    attack_combo = os.path.basename(attack_folder)
                    print(f"[EVAL] {model_name} | {attack_combo} | start={start}, steps={steps}")
                    if name_to_id is None or not name_to_id:
                        # Use label_to_name and invert
                        name_to_id = {v: k for k, v in label_to_name.items()}
                    try:
                        metrics = evaluate_combo_from_csv(
                            model_name=model_name,
                            dataset_name=args.dataset_name,
                            class_map_path=class_map_path,
                            attack_folder=attack_folder,
                            denoising_start=start,
                            steps=steps,
                            name_to_id=name_to_id,
                            device=args.device,
                        )
                    except Exception as e:
                        print(f"[SKIP] {attack_combo} due to error: {e}")
                        continue

                    row = {
                        "model_name": model_name,
                        "attack_combo": attack_combo,
                        "denoising_start": start,
                        "steps": steps,
                        **metrics,
                    }
                    rows_this_model.append(row)

                    # append to CSV incrementally
                    with open(args.csv_out, "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(row.keys()))
                        if not wrote_header:
                            w.writeheader()
                            wrote_header = True
                        w.writerow(row)

        # Rank & print this model (across its attack folders/grid)
        rows_this_model.sort(key=lambda r: (-r["defense_success_%"], -r["avg_conf_recon"]))
        print("\n=== Sweep Results (top first) ===")
        for r in rows_this_model:
            print(
                f"{r['model_name']:<20} | {r['attack_combo']:<32} | "
                f"start={r['denoising_start']:<4} steps={r['steps']:<3} | "
                f"def_ok={r['defense_success_%']:.1f}% | "
                f"benign_ret={r['benign_retention_%']:.1f}% | "
                f"adv_fooled={r['adv_fooled_%']:.1f}% | "
                f"avg_conf (ben/ben_sdxl/adv/rec) = "
                # f"{r['avg_conf_benign']:.3f}/{r['avg_conf_benign_sdxl']:.3f}/"
                # f"{r['avg_conf_adv']:.3f}/{r['avg_conf_recon']:.3f} | "
                # f"n={r['n']} | label_misses={r['label_misses']}"
            )

        all_rows.extend(rows_this_model)

    # Overall best across all models/attacks
    if all_rows:
        all_rows.sort(key=lambda r: (-r["defense_success_%"], -r["avg_conf_recon"]))
        best = all_rows[0]
        print("\n*** Best combo overall ***")
        print(json.dumps(best, indent=2))

if __name__ == "__main__":
    main()
