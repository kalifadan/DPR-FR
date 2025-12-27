from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from PIL import Image


@dataclass
class SDXLPurifier:
    """
    SDXL img2img purifier. Designed as a drop-in front-end:
      List[PIL.Image] -> List[PIL.Image]

    Notes:
    - Uses empty prompt and guidance_scale=0 for "unconditional" cleanup-style behavior.
    - Uses config-style denoising_start; if pipeline doesn't accept it, maps to strength.
    - Runs in chunks of PURIFIER_BATCH_SIZE to avoid OOM.
    """

    model_id: str
    num_steps: int
    denoising_start: float
    resolution: int = 512
    batch_size: int = 1
    device: Optional[str] = None
    seed: int = 0

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device != "cuda":
            raise RuntimeError("SDXL purifier is intended for CUDA. Set PURIFIER_NAME='none' for CPU runs.")

        # Lazy import so your baseline still runs without diffusers installed
        try:
            from diffusers import StableDiffusionXLImg2ImgPipeline
        except Exception as e:
            raise ImportError(
                "diffusers is required for SDXL purification. Install: pip install diffusers transformers accelerate safetensors"
            ) from e

        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        # Speed/memory knobs (safe defaults)
        self.pipe.enable_attention_slicing()
        # self.pipe.enable_xformers_memory_efficient_attention()  # optional if you have xformers

        self.pipe = self.pipe.to(self.device)

        # Some pipelines add watermarking; you can disable if present
        if hasattr(self.pipe, "watermark") and self.pipe.watermark is not None:
            try:
                self.pipe.watermark = None
            except Exception:
                pass

    @property
    def signature(self) -> str:
        # Must uniquely define the purification settings for caching
        return (
            f"sdxl|model={self.model_id}|steps={self.num_steps}|"
            f"denstart={self.denoising_start}|res={self.resolution}|seed={self.seed}"
        )

    def _prep(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        # SDXL is trained at higher res; we use a fixed square for speed.
        return img.resize((self.resolution, self.resolution), Image.BILINEAR)

    def __call__(self, images: List[Image.Image]) -> List[Image.Image]:
        if len(images) == 0:
            return images

        from inspect import signature as py_signature

        out_images: List[Image.Image] = []
        gen = torch.Generator(device=self.device).manual_seed(self.seed)

        call_sig = py_signature(self.pipe.__call__)
        supports_denoising_start = "denoising_start" in call_sig.parameters

        # Map denoising_start -> strength if needed.
        # Intuition: later start => less modification => smaller strength.
        strength = float(max(0.0, min(1.0, 1.0 - self.denoising_start)))

        for i in range(0, len(images), self.batch_size):
            chunk = [self._prep(im) for im in images[i : i + self.batch_size]]

            kwargs = dict(
                prompt=[""] * len(chunk),
                image=chunk,
                guidance_scale=0.0,
                num_inference_steps=int(self.num_steps),
                generator=gen,
            )

            if supports_denoising_start:
                kwargs["denoising_start"] = float(self.denoising_start)
            else:
                kwargs["strength"] = strength

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                res = self.pipe(**kwargs)

            # diffusers returns images as res.images
            out_images.extend(res.images)

        return out_images
