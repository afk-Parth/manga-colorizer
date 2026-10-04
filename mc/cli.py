"""Command-line fallback:  python -m mc doctor | run INPUT -o OUT [--quick]"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from .bank import CharacterBank
from .config import CHAR_DIR, Settings
from .engine import Engine
from .io_utils import encode_png, pages_from_path
from .pipeline import analyse_page, colorize_page, prepare_characters


def doctor(a):
    print("Python:", sys.version.split()[0])
    for m in ("numpy", "cv2", "PIL", "streamlit", "torch", "transformers", "diffusers", "fitz"):
        try:
            mod = __import__(m)
            print(f"  {m:14s} OK  {getattr(mod, '__version__', '')}")
        except Exception as e:  # noqa: BLE001
            print(f"  {m:14s} MISSING ({type(e).__name__})")
    eng = Engine()
    print("Compute device:", eng.device)
    print("Characters:", CharacterBank(CHAR_DIR).names() or "none yet")
    if a.models:
        print("\nLoading models (first run downloads them)...")
        for key in ("detector", "identifier", "parser"):
            t = time.time()
            ok = getattr(eng, key) is not None
            print(f"  {key:10s} {'OK' if ok else 'FAILED: ' + eng.errors.get(key, '?')}  ({time.time() - t:.0f}s)")
        t = time.time()
        c, err = eng.get_colorizer("ai")
        print(f"  {'diffusion':10s} {'OK' if c else 'FAILED: ' + str(err)}  ({time.time() - t:.0f}s)")


def run(a):
    st = Settings(backend="quick" if a.quick else "ai", rtl=not a.ltr, low_memory=a.low_memory)
    eng = Engine(low_memory=a.low_memory)
    bank = CharacterBank(CHAR_DIR)
    colorizer, err = eng.get_colorizer(st.backend)
    if colorizer is None:
        sys.exit(f"Could not load the AI colouriser: {err}\nTry:  python -m mc run ... --quick   (no AI)")
    for m in prepare_characters(bank, eng, st):
        print("note:", m)
    always = bank.names() if len(bank.names()) == 1 else []
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    for name, gray in pages_from_path(a.input):
        t = time.time()
        an = analyse_page(name, gray, bank, eng, st, always)
        rgb, problems = colorize_page(gray, an, bank, eng, st, colorizer)
        (out / name).write_bytes(encode_png(rgb))
        print(f"{name}: {len(an.panels)} panels, {time.time() - t:.0f}s", ("  PROBLEMS: " + "; ".join(problems)) if problems else "")
    for k, v in eng.errors.items():
        print(f"warning [{k}]: {v}")


def main():
    ap = argparse.ArgumentParser("mc")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("doctor", help="check installation")
    d.add_argument("--models", action="store_true", help="also load/download all AI models")
    d.set_defaults(fn=doctor)
    r = sub.add_parser("run", help="colourise without the UI")
    r.add_argument("input")
    r.add_argument("-o", "--output", default="workspace/output")
    r.add_argument("--quick", action="store_true", help="no AI, flat colours")
    r.add_argument("--ltr", action="store_true")
    r.add_argument("--low-memory", action="store_true")
    r.set_defaults(fn=run)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
