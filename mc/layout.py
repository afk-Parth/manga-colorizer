"""Panel + speech-bubble detection (classic OpenCV heuristics, no model download)."""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]  # x, y, w, h


def _drop_contained(boxes: List[Box], frac: float = 0.7) -> List[Box]:
    keep = []
    for i, (x, y, w, h) in enumerate(boxes):
        inside = False
        for j, (X, Y, W, H) in enumerate(boxes):
            if i == j or W * H < w * h or (W * H == w * h and j > i):
                continue
            ix = max(0, min(x + w, X + W) - max(x, X))
            iy = max(0, min(y + h, Y + H) - max(y, Y))
            if ix * iy > frac * w * h:
                inside = True
                break
        if not inside:
            keep.append((x, y, w, h))
    return keep


def detect_panels(gray: np.ndarray, rtl: bool = True, min_area_frac: float = 0.02) -> List[Box]:
    h, w = gray.shape
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = max(5, int(min(h, w) * 0.01)) | 1
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    cnts, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [tuple(int(v) for v in cv2.boundingRect(c)) for c in cnts]
    boxes = [b for b in boxes if b[2] * b[3] >= min_area_frac * h * w]
    if not boxes:
        return [(0, 0, w, h)]
    boxes = _drop_contained(boxes)
    row_h = 0.15 * h
    boxes.sort(key=lambda b: (round(b[1] / row_h), -b[0] if rtl else b[0]))
    return boxes


def bubble_mask(gray: np.ndarray) -> np.ndarray:
    """Speech bubbles = small, convex white blobs containing several letter-sized dark specks."""
    h, w = gray.shape
    page = h * w
    _, white = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=4)
    mask = np.zeros_like(gray)
    for i in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[i]]
        if area < 0.003 * page or area > 0.06 * page:
            continue
        if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            continue
        if max(bw, bh) / max(1, min(bw, bh)) > 4:
            continue
        comp = (lab[y:y + bh, x:x + bw] == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        hull = cv2.contourArea(cv2.convexHull(c))
        if hull == 0 or cv2.contourArea(c) / hull < 0.9:
            continue
        filled = np.zeros_like(comp)
        cv2.drawContours(filled, [c], -1, 1, -1)
        holes = ((filled > 0) & (comp == 0)).astype(np.uint8)
        _, _, hs, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
        specks = [a for a in hs[1:, cv2.CC_STAT_AREA] if 4 <= a <= 0.03 * filled.sum()]
        frac = holes.sum() / max(1, filled.sum())
        if len(specks) >= 3 and 0.02 < frac < 0.35:
            mask[y:y + bh, x:x + bw][filled > 0] = 255
    return cv2.dilate(mask, np.ones((5, 5), np.uint8))
