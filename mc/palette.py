from __future__ import annotations

import colorsys
from typing import List, Sequence, Tuple

import cv2
import numpy as np

RGB = Tuple[int, int, int]


def rgb_to_hex(c: Sequence[int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*[int(max(0, min(255, v))) for v in c])


def hex_to_rgb(h: str) -> RGB:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def rgb_to_lab(c: Sequence[int]) -> np.ndarray:
    a = np.array(c, np.float32).reshape(1, 1, 3) / 255.0
    return cv2.cvtColor(a, cv2.COLOR_RGB2LAB).reshape(3)


def extract_palette(rgb_images: List[np.ndarray], k: int = 6) -> List[RGB]:
    """k-means in Lab over reference images, ignoring near-white background."""
    px = []
    for im in rgb_images:
        small = cv2.resize(im, (128, max(1, int(128 * im.shape[0] / im.shape[1]))))
        lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).reshape(-1, 3)
        px.append(lab[lab[:, 0] < 240])
    if not px:
        return []
    px = np.concatenate(px).astype(np.float32)
    if len(px) < k:
        return []
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centers = cv2.kmeans(px, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    order = np.argsort(-np.bincount(labels.ravel(), minlength=k))
    centers = centers[order].astype(np.uint8).reshape(1, -1, 3)
    return [tuple(int(v) for v in c) for c in cv2.cvtColor(centers, cv2.COLOR_LAB2RGB).reshape(-1, 3)]


def enforce_palette(rgb: np.ndarray, palette: List[RGB], strength: float = 0.6, thresh: float = 40.0) -> np.ndarray:
    """Pull colours that are close to a stored palette colour toward it (Lab space)."""
    if not palette or strength <= 0:
        return rgb
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    pal = cv2.cvtColor(np.array(palette, np.float32).reshape(1, -1, 3) / 255.0, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    out = lab.copy()
    for y in range(0, lab.shape[0], 128):
        blk = lab[y:y + 128]
        d = np.linalg.norm(blk[:, :, None, :] - pal[None, None, :, :], axis=3)
        idx, dmin = d.argmin(2), d.min(2)
        wgt = np.clip(1 - dmin / thresh, 0, 1) * strength
        sh = (pal[idx] - blk) * wgt[..., None]
        sh[..., 0] *= 0.5  # keep most of the original lightness (shading)
        out[y:y + 128] = blk + sh
    return np.clip(cv2.cvtColor(out, cv2.COLOR_LAB2RGB) * 255, 0, 255).astype(np.uint8)


def apply_lines(rgb: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Re-impose the original ink and tone so linework stays crisp."""
    g = (gray.astype(np.float32) / 255.0)[..., None]
    return np.clip(rgb.astype(np.float32) * (0.35 + 0.65 * g), 0, 255).astype(np.uint8)


def color_name(c: Sequence[int]) -> str:
    r, g, b = [v / 255.0 for v in c]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h *= 360
    if v < 0.18:
        return "black"
    if s < 0.12:
        return "white" if v > 0.85 else "light gray" if v > 0.6 else "gray" if v > 0.35 else "dark gray"
    if h < 15 or h >= 345:
        name = "red"
    elif h < 40:
        name = "brown" if v < 0.6 else "orange"
    elif h < 65:
        name = "yellow"
    elif h < 165:
        name = "green"
    elif h < 200:
        name = "teal"
    elif h < 260:
        name = "blue"
    elif h < 300:
        name = "purple"
    else:
        name = "pink"
    if name != "brown" and v < 0.4:
        return "dark " + name
    if v > 0.85 and s < 0.35:
        return "light " + name
    return name
