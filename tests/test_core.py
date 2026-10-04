"""Run:  python tests/test_core.py     (no torch / GPU needed - AI models are replaced by fakes)"""
import sys, tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mc.bank import CharacterBank, GROUPS
from mc.config import Settings
from mc.consistency import detect_parts, enforce, part_shift
from mc.engine import Engine
from mc.io_utils import decode_gray, encode_png, pages_from_bytes
from mc.layout import bubble_mask, detect_panels
from mc.models import QuickColorizer
from mc.palette import color_name, hex_to_rgb, rgb_to_hex
from mc.pipeline import analyse_page, colorize_page, prepare_characters
from mc.viz import draw_overlay


def synthetic_page():
    pg = np.full((1200, 800), 255, np.uint8)
    for (x, y, w, h) in [(40, 40, 720, 520), (40, 620, 340, 540), (420, 620, 340, 540)]:
        cv2.rectangle(pg, (x, y), (x + w, y + h), 0, 4)
    cv2.circle(pg, (400, 330), 130, 200, -1); cv2.circle(pg, (400, 330), 130, 0, 3)
    cv2.ellipse(pg, (620, 130), (110, 60), 0, 0, 360, 255, -1); cv2.ellipse(pg, (620, 130), (110, 60), 0, 0, 360, 0, 3)
    cv2.putText(pg, "HI! OK", (550, 145), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 3)
    return pg


def ref_image():
    r = np.full((300, 200, 3), 255, np.uint8)
    r[0:100] = (30, 25, 25)        # hair  - black
    r[100:200] = (240, 235, 235)   # top   - white
    r[200:300] = (200, 60, 60)     # skirt - red
    return r


class FakeParser:  # top third = hair, middle = top, bottom = skirt
    def parse(self, rgb):
        h = rgb.shape[0]
        m = np.zeros(rgb.shape[:2], np.uint8)
        m[: h // 3] = GROUPS.index("hair") + 1
        m[h // 3: 2 * h // 3] = GROUPS.index("top") + 1
        m[2 * h // 3:] = GROUPS.index("skirt") + 1
        return m


class FakeDetector:
    def detect(self, gray, thr):
        h, w = gray.shape
        return [(int(w * .2), int(h * .2), int(w * .5), int(h * .7), 0.9)]


class FakeIdent:
    def match(self, crops, bank):
        return [("hero", 0.9)] * len(crops)


def test():
    tmp = Path(tempfile.mkdtemp())
    # --- io
    page = synthetic_page()
    assert decode_gray(encode_png(page)).shape == page.shape
    assert pages_from_bytes("a.png", encode_png(page))[0][0] == "a.png"
    try:
        pages_from_bytes("bad.png", b"notanimage"); raise SystemExit("should fail")
    except RuntimeError:
        pass
    # --- layout
    panels = detect_panels(page)
    assert len(panels) == 3, panels
    assert panels[0][1] < 100 and panels[1][0] > panels[2][0], panels   # right-to-left order
    assert (bubble_mask(page) > 0).sum() > 1000
    # --- bank
    bank = CharacterBank(tmp / "chars")
    c = bank.add("Hero Girl!", [("r.png", encode_png(ref_image()))], outfit="school uniform")
    assert c.name == "Hero_Girl" and len(c.palette) == 6
    assert "school uniform" in c.prompt()
    try:
        bank.add("x", [("bad.png", b"zzz")]); raise SystemExit("should fail")
    except ValueError:
        pass
    assert "x" not in bank.names()
    # --- parts detection + persistence
    c.name = "hero"
    parts = detect_parts(c, FakeParser())
    assert set(parts) == {"hair", "top", "skirt"}, parts
    assert color_name(parts["hair"]) == "black" and color_name(parts["skirt"]) == "red", parts
    c.parts = parts; c.save_meta()
    c2 = CharacterBank(tmp / "chars").chars["Hero_Girl"]
    assert c2.parts["skirt"] == parts["skirt"] and "red skirt" in c2.prompt()
    # --- consistency correction pulls a wrong (blue) skirt toward red
    fake_out = np.zeros((300, 200, 3), np.uint8)
    fake_out[:100] = (30, 25, 25); fake_out[100:200] = (240, 235, 235); fake_out[200:] = (60, 60, 200)
    fixed = part_shift(fake_out, FakeParser(), parts, 1.0)
    assert fixed[250, 100, 0] > fake_out[250, 100, 0] + 20, (fixed[250, 100], fake_out[250, 100])
    # --- full pipeline with fake models, quick colouriser
    eng = Engine()
    eng.inject("detector", FakeDetector()); eng.inject("identifier", FakeIdent()); eng.inject("parser", FakeParser())
    bank = CharacterBank(tmp / "chars")
    bank.chars["Hero_Girl"].name = "hero"; bank.chars["hero"] = bank.chars.pop("Hero_Girl")
    st = Settings(backend="quick")
    an = analyse_page("p.png", page, bank, eng, st)
    assert len(an.panels) == 3 and all(p.dets and p.assigned == ["hero"] for p in an.panels), an
    rgb, problems = colorize_page(page, an, bank, eng, st, QuickColorizer())
    assert rgb.shape == page.shape + (3,) and not problems, problems
    bm = bubble_mask(page) > 0
    assert np.array_equal(rgb[bm][:, 0], page[bm])           # bubbles untouched
    assert (rgb.astype(int).std(axis=2) > 5).sum() > 5000     # something got coloured
    assert draw_overlay(page, an).shape == rgb.shape
    # --- a failing panel must not break the page
    calls = {"n": 0}
    def flaky(crop, chars, st):
        calls["n"] += 1
        if calls["n"] == 2: raise RuntimeError("boom (e.g. out of memory)")
        return QuickColorizer()(crop, chars, st)
    rgb, problems = colorize_page(page, an, bank, eng, st, flaky)
    assert len(problems) == 1 and "boom" in problems[0]
    # --- manual 'always' characters when nothing is detected
    eng2 = Engine(); eng2.inject("detector", type("D", (), {"detect": lambda s, g, t: []})())
    an2 = analyse_page("p.png", page, bank, eng2, st, always=["hero"])
    assert all(p.assigned == ["hero"] for p in an2.panels)
    # --- no AI libraries installed -> degrade gracefully (this sandbox has no torch)
    eng3 = Engine()
    an3 = analyse_page("p.png", page, bank, eng3, st)
    assert len(an3.panels) == 3 and all(not p.dets for p in an3.panels)
    msgs = prepare_characters(bank, eng3, Settings())
    colorizer, err = eng3.get_colorizer("ai")
    assert colorizer is None and err, "AI colouriser should report a clear error, not crash"
    assert eng3.get_colorizer("quick")[0] is not None
    # --- hex helpers
    assert hex_to_rgb(rgb_to_hex((1, 2, 3))) == (1, 2, 3)
    # --- real page sample (if available locally)
    real = Path("/mnt/user-data/uploads/comic-book-page-with-panels-anime-style-depicts-30-years-old-might-worker-dress_1097265-53338_jpg.avif")
    if real.exists():
        g = decode_gray(real.read_bytes())
        an4 = analyse_page("real.png", g, bank, Engine(), st, always=["hero"])
        rgb4, pr = colorize_page(g, an4, bank, Engine(), st, QuickColorizer())
        assert rgb4.shape[:2] == g.shape and not pr, pr
        print("real page OK:", len(an4.panels), "panels")
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    test()
