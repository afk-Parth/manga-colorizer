from __future__ import annotations

import io
import re
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif"}


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def decode_gray(buf: bytes) -> Optional[np.ndarray]:
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        try:
            from PIL import Image
            img = np.array(Image.open(io.BytesIO(buf)).convert("L"))
        except Exception:
            return None
    return img


def decode_rgb(buf: bytes) -> Optional[np.ndarray]:
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    if img is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        from PIL import Image
        return np.array(Image.open(io.BytesIO(buf)).convert("RGB"))
    except Exception:
        return None


def encode_png(img: np.ndarray) -> bytes:
    """img: RGB (H,W,3) or gray (H,W)."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


def pages_from_bytes(filename: str, data: bytes) -> List[Tuple[str, np.ndarray]]:
    """Return [(page_name, gray_page)] from an image, .pdf or .cbz/.zip upload."""
    ext = Path(filename).suffix.lower()
    stem = Path(filename).stem
    if ext == ".pdf":
        try:
            import fitz  # pymupdf
        except ImportError:
            raise RuntimeError("PDF support needs pymupdf:  pip install pymupdf")
        doc = fitz.open(stream=data, filetype="pdf")
        out = []
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), colorspace=fitz.csGRAY)
            img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()
            out.append((f"{stem}_p{i + 1:03d}.png", img))
        return out
    if ext in (".cbz", ".zip"):
        out = []
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = sorted((n for n in z.namelist() if Path(n).suffix.lower() in IMG_EXT), key=natural_key)
            for n in names:
                img = decode_gray(z.read(n))
                if img is not None:
                    out.append((Path(n).stem + ".png", img))
        if not out:
            raise RuntimeError("No readable images inside the archive")
        return out
    img = decode_gray(data)
    if img is None:
        raise RuntimeError(f"Could not read image '{filename}' (unsupported or corrupt). Try converting it to PNG/JPG.")
    return [(stem + ".png", img)]


def pages_from_path(path: str) -> List[Tuple[str, np.ndarray]]:
    p = Path(path)
    if p.is_dir():
        out = []
        for f in sorted((f for f in p.iterdir() if f.suffix.lower() in IMG_EXT), key=lambda f: natural_key(f.name)):
            try:
                out.extend(pages_from_bytes(f.name, f.read_bytes()))
            except Exception as e:  # skip unreadable files
                print(f"skip {f.name}: {e}")
        return out
    return pages_from_bytes(p.name, p.read_bytes())
