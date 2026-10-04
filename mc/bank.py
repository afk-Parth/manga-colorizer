"""Character memory bank. Each character = a folder with reference images + meta.json."""
from __future__ import annotations

import json
import re
import shutil
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .io_utils import IMG_EXT, decode_rgb
from .palette import RGB, color_name, extract_palette

GROUPS = ["hair", "skin", "top", "skirt", "pants", "dress", "shoes", "bag", "hat", "scarf", "belt"]
PROMPT_GROUPS = ["hair", "top", "skirt", "pants", "dress", "shoes", "bag", "hat", "scarf"]


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", name.strip()).strip("_")


class Character:
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.name = self.folder.name
        self.outfit = ""
        self.parts: Dict[str, RGB] = {}
        self.parts_tried = False
        self._palette: Optional[List[RGB]] = None
        self.reload()

    def reload(self):
        files = sorted(f for f in self.folder.iterdir() if f.suffix.lower() in IMG_EXT)
        self.ref_files = files
        self.refs: List[np.ndarray] = []
        for f in files:
            im = decode_rgb(f.read_bytes())
            if im is not None:
                self.refs.append(im)
        meta = self.folder / "meta.json"
        if meta.exists():
            try:
                d = json.loads(meta.read_text())
                self.outfit = d.get("outfit", "")
                self.parts = {k: tuple(v) for k, v in d.get("parts", {}).items() if k in GROUPS}
            except Exception:
                pass
        self._palette = None

    def save_meta(self):
        (self.folder / "meta.json").write_text(json.dumps({"outfit": self.outfit, "parts": {k: list(v) for k, v in self.parts.items()}}, indent=2))

    @property
    def palette(self) -> List[RGB]:
        if self._palette is None:
            self._palette = extract_palette(self.refs)
        return self._palette

    @property
    def seed(self) -> int:
        return zlib.crc32(self.name.encode())

    def signature(self) -> str:
        return "|".join(f"{f.name}:{f.stat().st_mtime_ns}" for f in self.ref_files)

    def prompt(self) -> str:
        bits = [f"{color_name(self.parts[g])} {g}" for g in PROMPT_GROUPS if g in self.parts]
        if self.outfit:
            bits.append(self.outfit)
        return ", ".join(bits)


class CharacterBank:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.chars: Dict[str, Character] = {}
        self.reload()

    def reload(self):
        self.chars = {}
        for d in sorted(p for p in self.root.iterdir() if p.is_dir()):
            if any(f.suffix.lower() in IMG_EXT for f in d.iterdir()):
                try:
                    c = Character(d)
                    if c.refs:
                        self.chars[c.name] = c
                except Exception:
                    continue

    def names(self) -> List[str]:
        return list(self.chars)

    def add(self, name: str, files: List[Tuple[str, bytes]], outfit: str = "") -> Character:
        name = safe_name(name)
        if not name:
            raise ValueError("Please give the character a name")
        folder = self.root / name
        folder.mkdir(parents=True, exist_ok=True)
        n = len([f for f in folder.iterdir() if f.suffix.lower() in IMG_EXT])
        saved = 0
        for fname, data in files:
            im = decode_rgb(data)
            if im is None:
                continue
            h, w = im.shape[:2]
            s = 768 / max(h, w)
            if s < 1:
                im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            n += 1
            cv2.imwrite(str(folder / f"ref_{n:02d}.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            saved += 1
        if saved == 0:
            if not any(f.suffix.lower() in IMG_EXT for f in folder.iterdir()):
                shutil.rmtree(folder, ignore_errors=True)
            raise ValueError("None of the uploaded reference images could be read")
        c = Character(folder)
        if outfit:
            c.outfit = outfit
            c.save_meta()
        self.chars[name] = c
        return c

    def delete(self, name: str):
        shutil.rmtree(self.root / name, ignore_errors=True)
        self.chars.pop(name, None)
