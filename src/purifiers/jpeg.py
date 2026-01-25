# DPR-FR/src/purifiers/jpeg.py
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import List
from PIL import Image


@dataclass
class JPEGPurifier:
    """
    JPEG-compression purifier (test-time only).
    List[PIL.Image] -> List[PIL.Image]
    """
    quality: int = 75          # 1..95 typical; lower = stronger compression
    subsampling: int = 2       # 0=4:4:4, 1=4:2:2, 2=4:2:0 (PIL default)
    optimize: bool = True

    @property
    def signature(self) -> str:
        return f"jpeg|q={self.quality}|sub={self.subsampling}|opt={int(self.optimize)}"

    def __call__(self, images: List[Image.Image]) -> List[Image.Image]:
        out = []
        for im in images:
            im = im.convert("RGB")
            buf = BytesIO()
            im.save(
                buf,
                format="JPEG",
                quality=int(self.quality),
                subsampling=int(self.subsampling),
                optimize=bool(self.optimize),
            )
            buf.seek(0)
            out.append(Image.open(buf).convert("RGB"))
        return out
