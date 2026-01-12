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


def get_purifier(name: str, model_id: str, num_steps: int = 4, denoising_start: float = 0.6, num_variants: int = 1,
                 device: str = "cuda", seed: int = 0):
    name = (name or "none").lower()
    if name in {"none", "noop"}:
        return NoOpPurifier()

    if name == "sdxl":
        from .sdxl import SDXLPurifier
        return SDXLPurifier(
            model_id=model_id,
            num_steps=num_steps,
            denoising_start=denoising_start,
            resolution=512,
            batch_size=1,
            device=device,
            seed=int(seed),
        )

    raise ValueError(f"Unknown PURIFIER_NAME={name}. Supported: 'none', 'sdxl'.")
