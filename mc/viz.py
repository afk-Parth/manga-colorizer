from __future__ import annotations

import cv2
import numpy as np

from .layout import bubble_mask


def draw_overlay(gray: np.ndarray, analysis) -> np.ndarray:
    """Panels (blue), detected characters (green = named, orange = unknown), speech bubbles (tinted)."""
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    h, w = gray.shape
    bm = bubble_mask(gray) > 0
    img[bm] = (0.55 * img[bm] + 0.45 * np.array([60, 200, 120])).astype(np.uint8)
    t = max(2, w // 350)
    fs = max(0.6, w / 900)
    for i, p in enumerate(analysis.panels):
        x, y, bw, bh = p.box
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (40, 90, 255), t * 2)
        label = f"{i + 1}: " + (", ".join(p.assigned) if p.assigned else "-")
        cv2.putText(img, label, (x + 10, y + int(40 * fs)), cv2.FONT_HERSHEY_SIMPLEX, fs, (40, 90, 255), t + 1, cv2.LINE_AA)
        for d in p.dets:
            dx, dy, dw, dh = d.box
            col = (0, 170, 0) if d.name else (255, 140, 0)
            cv2.rectangle(img, (dx, dy), (dx + dw, dy + dh), col, t)
            tag = f"{d.name or '?'}" + (f" {d.sim:.2f}" if d.sim is not None else "")
            cv2.putText(img, tag, (dx + 4, dy + dh - 8), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8, col, t, cv2.LINE_AA)
    return img
