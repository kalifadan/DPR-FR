from __future__ import annotations

from dataclasses import dataclass
from typing import List
from PIL import Image


class BasePurifier:
    """Callable purifier: List[PIL.Image] -> List[PIL.Image]."""

    @property
    def signature(self) -> str:
        """String that uniquely identifies this purifier configuration for caching."""
        raise NotImplementedError

    def __call__(self, images: List[Image.Image]) -> List[Image.Image]:
        raise NotImplementedError


@dataclass
class NoOpPurifier(BasePurifier):
    """No purification; returns inputs unchanged."""

    def __call__(self, images: List[Image.Image]) -> List[Image.Image]:
        return images

    @property
    def signature(self) -> str:
        return "none"


def get_purifier(name: str, num_steps: int = 4, denoising_start: float = 0.6, num_variants: int = 1):
    name = (name or "none").lower()
    if name in {"none", "noop"}:
        return NoOpPurifier()

    if name == "sdxl":
        from .sdxl import SDXLPurifier
        import config as _cfg
        return SDXLPurifier(
            model_id=_cfg.PURIFIER_MODEL_ID,
            num_steps=_cfg.PURIFIER_NUM_STEPS,
            denoising_start=_cfg.PURIFIER_DENOISING_START,
            resolution=getattr(_cfg, "PURIFIER_RESOLUTION", 512),
            batch_size=getattr(_cfg, "PURIFIER_BATCH_SIZE", 1),
            device=getattr(_cfg, "DEVICE", "cuda"),
            seed=getattr(_cfg, "SEED", 0),
        )

    raise ValueError(f"Unknown PURIFIER_NAME={name}. Supported: 'none', 'sdxl'.")
