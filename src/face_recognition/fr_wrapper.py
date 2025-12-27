from __future__ import annotations

from typing import List, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class FacialRecognitionWrapper(nn.Module):
    """
    Minimal face-embedding wrapper.

    - Input: List[PIL.Image] or a single PIL.Image
    - Output: torch.Tensor of shape (B, D), L2-normalized

    Backbone: facenet-pytorch InceptionResnetV1 pretrained on VGGFace2 (default).
    This is a strong, simple baseline for embeddings on LFW-style images.
    """

    def __init__(
        self,
        backbone_name: str = "facenet_inceptionresnetv1_vggface2",
        return_mode: str = "embeddings",
        device: Optional[str] = None,
        image_size: int = 160,
    ):
        super().__init__()

        if return_mode != "embeddings":
            raise ValueError("This baseline wrapper currently supports return_mode='embeddings' only.")

        self.backbone_name = backbone_name
        self.return_mode = return_mode
        self.image_size = image_size

        # Lazy import so error message is clear if package missing
        try:
            from facenet_pytorch import InceptionResnetV1
        except Exception as e:
            raise ImportError(
                "facenet-pytorch is required. Install it with: pip install facenet-pytorch"
            ) from e

        if backbone_name == "facenet_inceptionresnetv1_vggface2":
            self.model = InceptionResnetV1(pretrained="vggface2").eval()
        elif backbone_name == "facenet_inceptionresnetv1_casia":
            self.model = InceptionResnetV1(pretrained="casia-webface").eval()
        else:
            raise ValueError(
                f"Unknown backbone_name={backbone_name}. "
                "Use 'facenet_inceptionresnetv1_vggface2' or 'facenet_inceptionresnetv1_casia'."
            )

        # Device handling
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_str = device
        self.to(device)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if hasattr(self, "model"):
            self.model.to(*args, **kwargs)
        return self

    @staticmethod
    def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
        """
        Convert PIL RGB image to float tensor in [0,1], shape (3,H,W).
        """
        if img.mode != "RGB":
            img = img.convert("RGB")
        x = torch.from_numpy(__import__("numpy").array(img)).permute(2, 0, 1).float() / 255.0
        return x

    def _preprocess(self, imgs: List[Image.Image]) -> torch.Tensor:
        """
        Preprocess for InceptionResnetV1:
        - resize to (160,160)
        - scale to [-1, 1]
        """
        tensors = []
        for im in imgs:
            im = im.convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
            t = self._pil_to_tensor(im)  # [0,1]
            t = (t * 2.0) - 1.0          # [-1,1]
            tensors.append(t)

        batch = torch.stack(tensors, dim=0)  # (B,3,H,W)
        return batch

    @torch.no_grad()
    def forward(self, images: Union[List[Image.Image], Image.Image]) -> torch.Tensor:
        if isinstance(images, Image.Image):
            images = [images]

        x = self._preprocess(images).to(self.device_str)
        emb = self.model(x)  # (B, D)
        emb = F.normalize(emb, dim=-1)
        return emb
