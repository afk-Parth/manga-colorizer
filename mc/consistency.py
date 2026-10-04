"""Colour-consistency: per-character hair / skin / clothes colours, enforced after colouring."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .bank import GROUPS, Character
from .palette import RGB, enforce_palette, rgb_to_lab


def detect_parts(char: Character, parser, min_frac: float = 0.004) -> Dict[str, RGB]:
    """Run the clothes parser on the character's coloured references -> median colour per part."""
    acc: Dict[str, list] = {g: [] for g in GROUPS}
    for ref in char.refs:
        h, w = ref.shape[:2]
        gmap = parser.parse(ref)
        lab = cv2.cvtColor(ref.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
        for gi, g in enumerate(GROUPS, start=1):
            m = (gmap == gi).astype(np.uint8)
            if m.sum() < min_frac * h * w:
                continue
            m = cv2.erode(m, np.ones((5, 5), np.uint8)) > 0   # avoid outline pixels at part borders
            if m.sum() < 30:
                continue
            acc[g].append(lab[m])
    parts: Dict[str, RGB] = {}
    for g, v in acc.items():
        if not v:
            continue
        med = np.median(np.concatenate(v), axis=0).astype(np.float32).reshape(1, 1, 3)
        rgb = cv2.cvtColor(med, cv2.COLOR_LAB2RGB).reshape(3)
        parts[g] = tuple(int(np.clip(c * 255, 0, 255)) for c in rgb)
    return parts


def part_shift(region: np.ndarray, parser, parts: Dict[str, RGB], strength: float) -> np.ndarray:
    """Parse the coloured region, then move each part's colour toward the character's stored colour."""
    gmap = parser.parse(region)
    lab = cv2.cvtColor(region.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    changed = False
    for gi, g in enumerate(GROUPS, start=1):
        if g not in parts:
            continue
        m = gmap == gi
        if m.sum() < 80:
            continue
        cur = np.median(lab[m], axis=0)
        delta = rgb_to_lab(parts[g]) - cur
        delta[0] *= 0.4                                  # mostly fix hue/chroma, keep shading
        norm = float(np.linalg.norm(delta))
        if norm > 45:                                    # never make wild corrections
            delta *= 45.0 / norm
        w = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.5)[..., None] * strength
        lab = lab + w * delta
        changed = True
    if not changed:
        raise RuntimeError("no parts found")
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB) * 255, 0, 255).astype(np.uint8)


def enforce(rgb: np.ndarray, local_dets: List[Tuple[str, Tuple[int, int, int, int]]],
            chars: Dict[str, Character], parser, strength: float) -> np.ndarray:
    """rgb: coloured panel. local_dets: [(character_name, (x,y,w,h) in panel coords)]."""
    if strength <= 0 or not chars:
        return rgb
    out = rgb.copy()
    h, w = out.shape[:2]
    targets = [(n, b) for n, b in local_dets if n in chars]
    if not targets and len(chars) == 1:
        targets = [(next(iter(chars)), (0, 0, w, h))]
    for name, (x, y, bw, bh) in targets:
        x, y = max(0, x), max(0, y)
        region = out[y:y + bh, x:x + bw]
        if region.shape[0] < 16 or region.shape[1] < 16:
            continue
        ch = chars[name]
        done = False
        if parser is not None and ch.parts:
            try:
                region[:] = part_shift(region, parser, ch.parts, strength)
                done = True
            except Exception:
                done = False
        if not done and ch.palette:
            region[:] = enforce_palette(region, ch.palette, strength)
    return out
