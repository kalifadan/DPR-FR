# DPR-FR/src/purifiers/smooth.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List
from PIL import Image, ImageFilter


@dataclass
class SmoothPurifier:
    """
    Simple smoothing purifier (test-time only).
    Supports: gaussian blur or median filter.
    """
    kind: str = "gaussian"     # "gaussian" or "median"
    radius: float = 1.0        # for gaussian
    median_size: int = 3       # for median (must be odd)

    @property
    def signature(self) -> str:
        return f"smooth|kind={self.kind}|r={self.radius}|m={self.median_size}"

    def __call__(self, images: List[Image.Image]) -> List[Image.Image]:
        out = []
        for im in images:
            im = im.convert("RGB")
            if self.kind == "gaussian":
                out.append(im.filter(ImageFilter.GaussianBlur(radius=float(self.radius))))
            elif self.kind == "median":
                out.append(im.filter(ImageFilter.MedianFilter(size=int(self.median_size))))
            else:
                raise ValueError(f"Unknown smooth kind: {self.kind}")
        return out
