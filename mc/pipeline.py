"""Analyse (panels -> characters) and colourise pages. Every optional stage degrades gracefully."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .bank import CharacterBank
from .config import Settings
from .consistency import detect_parts, enforce
from .layout import Box, bubble_mask, detect_panels
from .palette import apply_lines

MIN_PANEL = 48


@dataclass
class Det:
    name: Optional[str]
    box: Box                      # page coordinates
    score: float = 0.0
    sim: Optional[float] = None


@dataclass
class PanelInfo:
    box: Box
    dets: List[Det] = field(default_factory=list)
    assigned: List[str] = field(default_factory=list)


@dataclass
class PageAnalysis:
    name: str
    shape: Tuple[int, int]
    panels: List[PanelInfo]


def analyse_page(name: str, gray: np.ndarray, bank: CharacterBank, engine, st: Settings,
                 always: Sequence[str] = ()) -> PageAnalysis:
    panels = detect_panels(gray, rtl=st.rtl)
    detector = engine.detector if st.detect_characters else None
    ident = engine.identifier if (detector is not None and bank.chars) else None
    infos: List[PanelInfo] = []
    for (x, y, w, h) in panels:
        crop = gray[y:y + h, x:x + w]
        dets: List[Det] = []
        if detector is not None and min(w, h) >= MIN_PANEL:
            try:
                raw = detector.detect(crop, st.det_threshold)
            except Exception as e:  # noqa: BLE001
                engine.errors["detector_run"] = f"{type(e).__name__}: {e}"
                raw = []
            raw = [r for r in raw if r[2] * r[3] >= 0.02 * w * h]
            matches: List[Tuple[Optional[str], Optional[float]]] = [(None, None)] * len(raw)
            if ident is not None and raw:
                try:
                    matches = ident.match([crop[by:by + bh, bx:bx + bw] for bx, by, bw, bh, _ in raw], bank)
                except Exception as e:  # noqa: BLE001
                    engine.errors["identifier_run"] = f"{type(e).__name__}: {e}"
            for (bx, by, bw, bh, sc), (nm, sim) in zip(raw, matches):
                if nm is not None and (sim is None or sim < st.clip_threshold):
                    nm = None
                dets.append(Det(nm, (x + bx, y + by, bw, bh), sc, sim))
        assigned: List[str] = []
        for d in dets:
            if d.name and d.name not in assigned:
                assigned.append(d.name)
        for n in always:
            if n in bank.chars and n not in assigned:
                assigned.append(n)
        infos.append(PanelInfo((x, y, w, h), dets, assigned))
    return PageAnalysis(name, gray.shape[:2], infos)


def prepare_characters(bank: CharacterBank, engine, st: Settings) -> List[str]:
    """Auto-detect hair/skin/clothes colours for characters that have none yet. Returns messages."""
    msgs: List[str] = []
    if not st.use_parts:
        return msgs
    todo = [c for c in bank.chars.values() if not c.parts and not c.parts_tried]
    if not todo:
        return msgs
    parser = engine.parser
    if parser is None:
        msgs.append("Clothes parser unavailable (" + engine.errors.get("parser", "?") + ") - using palette matching instead.")
        return msgs
    for c in todo:
        c.parts_tried = True
        try:
            parts = detect_parts(c, parser)
            if parts:
                c.parts = parts
                c.save_meta()
        except Exception as e:  # noqa: BLE001
            msgs.append(f"{c.name}: could not detect clothes colours ({e})")
    return msgs


def colorize_page(gray: np.ndarray, analysis: PageAnalysis, bank: CharacterBank, engine, st: Settings,
                  colorizer, progress: Optional[Callable[[int, int, str], None]] = None):
    """Returns (rgb_page, problems[list of str])."""
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    parser = engine.parser if st.use_parts else None
    work = [p for p in analysis.panels if min(p.box[2], p.box[3]) >= MIN_PANEL]
    problems: List[str] = []
    for i, p in enumerate(work):
        x, y, w, h = p.box
        if progress:
            progress(i, len(work), f"panel {i + 1}/{len(work)}")
        crop = gray[y:y + h, x:x + w]
        chars = [bank.chars[n] for n in p.assigned if n in bank.chars]
        try:
            rgb = colorizer(crop, chars, st)
            local = [(d.name, (d.box[0] - x, d.box[1] - y, d.box[2], d.box[3])) for d in p.dets if d.name]
            rgb = enforce(rgb, local, {c.name: c for c in chars}, parser, st.enforce)
            canvas[y:y + h, x:x + w] = apply_lines(rgb, crop)
        except Exception as e:  # noqa: BLE001  (one bad panel must not kill the page)
            problems.append(f"panel {i + 1}: {type(e).__name__}: {e}")
    if progress:
        progress(len(work), len(work), "finishing")
    keep = bubble_mask(gray) > 0
    canvas[keep] = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)[keep]   # speech bubbles + lettering stay untouched
    return canvas, problems
