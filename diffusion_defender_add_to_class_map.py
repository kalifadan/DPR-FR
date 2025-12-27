from scripts.face_recognition.fr_wrapper import FacialRecognitionWrapper, FacialRecognitionClassMap
from scripts.face_recognition.fr_datasets import FRMoldDataset, collate_mold_fn
from torch.utils.data import DataLoader
from diffusers import StableDiffusionXLImg2ImgPipeline
from collections import defaultdict
import pickle
from typing import Dict
import torch
import torch.nn.functional as F
from travelers.data_molds import ImageMold, TensorableMold
import cv2
import numpy as np
import os
from PIL import Image

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

pipe.safety_checker = None  # for research


def mild_sharpen(pil_img):
    img = np.array(pil_img)
    blurred = cv2.GaussianBlur(img, (0,0), 0.5)
    sharp = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    return Image.fromarray(np.clip(sharp, 0, 255).astype(np.uint8))


def sdxl_purify(image: Image.Image, denoising_start: float = 0.5, steps: int = 20):
    """
    Use SDXL image-to-image diffusion to purify adversarial noise.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    with torch.no_grad():
        purified = pipe(
            prompt="",                # unconditional guidance
            image=image,
            # strength=strength,        # how much noise to add
            denoising_start = denoising_start, # start denoising at % of the way through
            num_inference_steps=steps,
            guidance_scale=0.0,       # disable text conditioning
            dtype=torch.float32
        ).images[0]
    return purified

# build_class_map
@torch.inference_mode()
def build_class_map(
    model: FacialRecognitionWrapper,              # on device, return_mode="embedding"
    dataloader: DataLoader,
    label_to_name: Dict[int, str],
    defended_class_map: bool = True,
    denoising_start: float = 0.9,
    steps: int = 10
) -> FacialRecognitionClassMap:

    """
    Build a class map for face recognition from the model and dataloader.
    Args:
        model (FacialRecognitionWrapper): Model to extract embeddings.
        dataloader (DataLoader): DataLoader with images and labels.
        label_to_name (Dict[int, str]): Mapping from class ID to class name.
        defended_class_map (bool): Whether to use defended class map with diffuser.
    Returns:
        FacialRecognitionClassMap: Class map containing class IDs, embeddings, names, and embedding dimension.
    """

    class_to_embeddings: Dict[int, list[torch.Tensor]] = defaultdict(list)

    for batch in dataloader:
        if batch is None:
            continue

        # lists from your collate
        images_list = batch["images"]   # list[ImageMold]
        labels_list = batch["labels"]   # list[TensorableMold] or list[Tensor]

        if defended_class_map:
            # Use diffuser to defend against adversarial attacks
            diffused_images_list = ImageMold([
                sdxl_purify(img.astype('PIL')[0], denoising_start=denoising_start, steps=steps) if defended_class_map else img
                for img in images_list
            ])
            
        # add both original and diffused images to the list
        total_images_list = ImageMold(images_list.astype('pil') + diffused_images_list.astype('pil'))
            
        # Forward: wrapper returns TensorableMold with (B, D)
        emb_mold: TensorableMold = model(total_images_list)
        embeddings: torch.Tensor = emb_mold.astype('torch',batched=True)         # (B, D) on device
        embeddings = embeddings.detach().cpu()

        # L2-normalize embeddings before averaging (ArcFace/CosFace assume cosine space).
        embeddings = F.normalize(embeddings, dim=-1)

        labels = labels_list.astype('torch',batched=True).repeat(2).detach().cpu()
        # Accumulate per class
        for emb, lbl in zip(embeddings, labels):
            class_to_embeddings[int(lbl.item())].append(emb)

    # Build embeddings prototypes
    ids, names, protos = [], [], []
    for lbl in sorted(class_to_embeddings.keys()):
        embs = torch.stack(class_to_embeddings[lbl], dim=0)   # (N_i, D) normalized
        mean_emb = embs.mean(dim=0)                           # (D,), take the mean of all embeddings an identity
        mean_emb = F.normalize(mean_emb, dim=0)               # Re-normalize the prototype after averaging (so each class vector is unit length).
        protos.append(mean_emb)
        ids.append(lbl)
        names.append(label_to_name.get(lbl, f"class_{lbl}"))

    if len(protos) == 0:
        raise ValueError("No embeddings collected. Check your dataloader/labels.")
    final_embeddings = torch.stack(protos, dim=0)

    return FacialRecognitionClassMap(
        ids=ids, # (num_classes,) - class IDs
        embeddings=final_embeddings,  # (num_classes, D) - class embeddings
        names=names, # (num_classes,) - class names
        embedding_dim=final_embeddings.shape[1]  # D - embedding dimension
    )


# ------------------------
# 4. Save & Load Class Map
# ------------------------
def save_class_map(class_map: FacialRecognitionClassMap, path: str):
    # Create the parent directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(class_map.to_dict(), f)


def load_class_map(path: str) -> FacialRecognitionClassMap:
    with open(path, "rb") as f:
        data = pickle.load(f)
        return FacialRecognitionClassMap.from_dict(data)


# ------------------------
# 5. Prepare Face Recognition Pipeline
# ------------------------
def prepare_face_recognition_pipeline(model_name: str, dataset_name: str, device: torch.device, defended_class_map: bool = False):
    """
    Prepare model, dataloader, and label-to-name mapping for face recognition.

    Args:
        model_name (str): Name of the model (e.g., "ms1mv3_arceface_iresnet100").
        dataset_name (str): Dataset name (currently only "lfw" and "DeepKeep_Face_Dataset" supported).
        device (torch.device): Device to load the model onto.
        defended_class_map (bool): Whether to use defended class map with diffuser.

    Returns:
        tuple: (model, dataloader, label_to_name)
    """

    # check if pickle file exists
    class_map_path = f"/home/me/fr_classmap/{model_name}_on_{dataset_name}_class_map.pkl" if not defended_class_map else f"/home/me/fr_classmap/{model_name}_on_{dataset_name}_defended_0.8_20_class_map.pkl"
    if os.path.exists(class_map_path):
        print(f"Class map already exists at {class_map_path}. Loading...")
        class_map = load_class_map(class_map_path)
        print(f"Loaded class map with {len(class_map.ids)} classes.")
        exit(0)
    # If not exists, build class map
    else:
        print(f"Building class map for {model_name} on {dataset_name} dataset...")

        # builad class map for arceface backbone and lfw dataset
        if dataset_name == "lfw":
            lfw_root = "/home/me/Datasets/LFW/lfw-deepfunneled/lfw-deepfunneled"
            # Check if class map for the first model and dataset already exists
            #if it is, we will want to use the class map id and name mapping to get the same mapping for the second model
            first_model_cls_path = f"/home/me/fr_classmap/ms1mv3_arceface_iresnet100_on_{dataset_name}_class_map.pkl" if not defended_class_map else f"/home/me/fr_classmap/ms1mv3_arceface_iresnet100_on_{dataset_name}_defended_class_map.pkl"
            if os.path.exists(first_model_cls_path):
                #avoid the embeddings of the first model, just use the id and name mapping
                dataset = FRMoldDataset(
                    root_dir=lfw_root,
                    transform=None,  # No transform needed for embeddings
                    class_map_paths=first_model_cls_path, #use the class map of the first model to get the same id and name mapping
                    ignore_embeddings = True,  # ensures we don’t use old embeddings
                )
            else:
             # normal loading
             dataset = FRMoldDataset(
                 root_dir=lfw_root,
                 transform=None,  # No transform needed for embeddings
             )
        elif dataset_name == "DeepKeep_Face_Dataset":
            deepkeep_root = "/home/me/Datasets/DeepKeep_Face_Dataset/images"
            # Check if class map for the first model and dataset already exists
            #if it is, we will want to use the class map id and name mapping to get the same mapping for the second model
            first_model_cls_path = f"/home/me/fr_classmap/ms1mv3_arceface_iresnet100_on_{dataset_name}_class_map.pkl" if not defended_class_map else f"/home/me/fr_classmap/ms1mv3_arceface_iresnet100_on_{dataset_name}_defended_class_map.pkl"
            if os.path.exists(first_model_cls_path):
                #avoid the embeddings of the first model, just use the id and name mapping
                dataset = FRMoldDataset(
                    root_dir=deepkeep_root,
                    transform=None,  # No transform needed for embeddings
                    class_map_paths=first_model_cls_path, #use the class map of the first model to get the same id and name mapping
                    ignore_embeddings = True,  # ensures we don’t use old embeddings
                )
            else:
                # normal loading
                dataset = FRMoldDataset(
                    root_dir=deepkeep_root,
                    transform=None,  # No transform needed for embeddings
                )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_mold_fn)

        # if model_name == "ms1mv3_arceface_iresnet100":
        #     model = iresnet100(pretrained=False)
        #     weight_path = "/home/me/FR_model_weights/ms1mv3_arcface_r100_fp16.pth"
        # elif model_name == "glint360k_cosface_iresnet100":
        #     model = iresnet100(pretrained=False)
        #     weight_path = "/home/me/FR_model_weights/glint360k_cosface_r100_fp16_0.1.pth"
        # else:
        #     raise ValueError(f"Unsupported model: {model_name}")
        #
        # checkpoint = torch.load(weight_path, map_location='cpu')
        # if 'state_dict' in checkpoint:
        #     checkpoint = checkpoint['state_dict']
        # # Load weights
        # model.load_state_dict(checkpoint, strict=False)
        # # to device
        # model = model.to(device)
        # # model eval
        # model.eval()
        #create fr wrapper
        fr_wrapper = FacialRecognitionWrapper(backbone_name=model_name, return_mode="embeddings")
        # build class map
        # dataset.label_map: {"George_W_Bush": 0, "Colin_Powell": 1, ...}
        label_to_name = {idx: name for name, idx in dataset.label_map.items()}

        return fr_wrapper, dataloader, label_to_name, class_map_path


#main
if __name__ == "__main__": 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name= "LVFace-ms1mv3_arceface_iresnet100"#"RepVGG_B1" #"glint360k_cosface_iresnet100" #"ReXNet_1"#"EfficientNet_B0"#"MobileFaceNet"#"Attention_92"#"Swin_S" #"RepVGG_B1" #"ms1mv3_arceface_iresnet100"
    dataset_name = "lfw"#"DeepKeep_Face_Dataset" #"lfw"
    #TODO: need to implement defended class map with diffuser for defence which is not existing yet
    defended_class_map = True  # whether to use defended class map with diffuser
    #create fr wrapper dataloader, label_to_name mapping and class_map_path for class map builder
    fr_wrapper, dataloader, label_to_name, class_map_path = prepare_face_recognition_pipeline(model_name=model_name,
                                                                                              dataset_name=dataset_name,
                                                                                              device=device,
                                                                                              defended_class_map =defended_class_map)
    
    denoising_start = 0.8
    steps = 20
    # Build class map for the model and dataset, which will be used for face recognition
    #class map will hold class ID, mean embedding and name -> per identity
    class_map = build_class_map(
        model=fr_wrapper,
        dataloader=dataloader,
        label_to_name=label_to_name,
        defended_class_map=defended_class_map,
        denoising_start=denoising_start,
        steps=steps
    )
    # Save class map with model name
    save_class_map(class_map, class_map_path)
    print(f"Class map saved to {class_map_path}")
