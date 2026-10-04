from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Must be set before torch is imported anywhere.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # Apple GPU: fall back to CPU for missing ops
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")     # slow connections: don't time out model downloads
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
CHAR_DIR = ROOT / "characters"
OUT_DIR = ROOT / "workspace" / "output"


def pick_device() -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Settings:
    backend: str = "ai"            # "ai" (diffusion) | "quick" (flat colours, instant)
    size: int = 640                # long side (px) of each panel fed to the diffusion model
    steps: int = 20
    guidance: float = 6.0
    ip_scale: float = 0.8          # how strongly the character reference image is followed
    control_scale: float = 1.0     # how strictly the line art is followed
    enforce: float = 0.6           # strength of the colour-consistency correction (0 = off)
    rtl: bool = True               # right-to-left reading order
    detect_characters: bool = True
    use_parts: bool = True         # clothes/hair/skin-level colour correction
    det_threshold: float = 0.08    # character detector sensitivity (lower = finds more)
    clip_threshold: float = 0.65   # similarity needed to auto-name a detected character
    low_memory: bool = False
