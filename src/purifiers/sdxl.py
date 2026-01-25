from __future__ import annotations

from dataclasses import dataclass, field
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
    - Supports config-style denoising_start; if pipeline doesn't accept it, maps to strength.
    - Runs in chunks of batch_size to avoid OOM.
    - IMPORTANT: we seed the RNG ONCE and reuse it across calls. Do NOT reseed per image/call,
      otherwise outputs can become nearly identical when you purify images one-by-one.
    """

    model_id: str
    num_steps: int
    denoising_start: float
    resolution: int = 512
    batch_size: int = 1
    device: Optional[str] = None
    seed: int = 0

    # Internal state (not part of the constructor signature)
    _gen: torch.Generator = field(init=False, repr=False)
    _supports_denoising_start: bool = field(init=False, repr=False)

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device != "cuda":
            raise RuntimeError(
                "SDXL purifier is intended for CUDA. Set PURIFIER_NAME='none' for CPU runs."
            )

        # Create a persistent generator ONCE (critical to avoid identical outputs per call)
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(int(self.seed))

        # Lazy import so your baseline still runs without diffusers installed
        try:
            from diffusers import StableDiffusionXLImg2ImgPipeline
        except Exception as e:
            raise ImportError(
                "diffusers is required for SDXL purification. Install: "
                "pip install diffusers transformers accelerate safetensors"
            ) from e

        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        try:
            self.pipe.vae.enable_slicing()
        except Exception:
            pass
        try:
            self.pipe.vae.enable_tiling()
        except Exception:
            pass

        try:
            self.pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass

        self.pipe = self.pipe.to(self.device)

        try:
            if hasattr(self.pipe, "vae") and self.pipe.vae is not None:
                # decode in fp32
                self.pipe.vae.to(dtype=torch.float32)
                # some diffusers versions use this flag
                if hasattr(self.pipe.vae, "config") and hasattr(self.pipe.vae.config, "force_upcast"):
                    self.pipe.vae.config.force_upcast = True
        except Exception:
            pass

        if hasattr(self.pipe, "watermark") and getattr(self.pipe, "watermark") is not None:
            try:
                self.pipe.watermark = None
            except Exception:
                pass

        # Cache whether denoising_start is supported by this diffusers version/pipeline
        try:
            from inspect import signature as py_signature

            call_sig = py_signature(self.pipe.__call__)
            self._supports_denoising_start = "denoising_start" in call_sig.parameters
        except Exception:
            self._supports_denoising_start = False

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

        out_images: List[Image.Image] = []

        # Map denoising_start -> strength if needed.
        # Intuition: later start => less modification => smaller strength.
        strength = float(max(0.0, min(1.0, 1.0 - float(self.denoising_start))))

        # Reuse the persistent generator so each call continues RNG state
        gen = self._gen

        for i in range(0, len(images), self.batch_size):
            chunk = [self._prep(im) for im in images[i : i + self.batch_size]]

            kwargs = dict(
                prompt=[""] * len(chunk),
                image=chunk,
                guidance_scale=0.0,
                num_inference_steps=int(self.num_steps),
                generator=gen,
            )

            strength = float(max(0.0, min(0.95, 1.0 - float(self.denoising_start))))
            kwargs["strength"] = strength

            with torch.inference_mode():
                res = self.pipe(**kwargs)

            # diffusers returns images as res.images
            out_images.extend(res.images)

        # Debug: detect collapse to constant images early
        try:
            if len(out_images) >= 2:
                import numpy as np
                a = np.asarray(out_images[0], dtype=np.int16)
                b = np.asarray(out_images[1], dtype=np.int16)
                mad = float(np.mean(np.abs(a - b)))
                if mad < 1e-3:
                    raise RuntimeError(
                        f"SDXL purifier collapse: outputs identical across inputs (MAD={mad:.6f}). "
                        f"Check VAE upcast + strength usage."
                    )
        except Exception:
            raise

        return out_images
