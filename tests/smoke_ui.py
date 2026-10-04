"""Runs app.py against a fake `streamlit` (real one isn't needed). Catches typos / wrong calls / crashes in every UI path.
   python tests/smoke_ui.py"""
import sys, tempfile, types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import test_core as T
from mc.io_utils import encode_png


class RerunSignal(Exception):
    pass


class F:  # fake uploaded file
    def __init__(self, name, data): self.name, self._d, self.size = name, data, len(data)
    def getvalue(self): return self._d


class Ctx:
    def __init__(self, fake): self.f = fake
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, n): return getattr(self.f, n)


class Fake:
    def __init__(self):
        self.session_state = {}
        self.pressed, self.radio_idx, self.uploads, self.texts, self.log, self._cache = set(), {}, {}, {}, [], {}
    # --- helpers
    def _hit(self, label): return any(p in str(label) for p in self.pressed)
    def _val(self, key, default):
        return self.session_state[key] if key is not None and key in self.session_state else default
    # --- layout
    def set_page_config(self, **k): pass
    def title(self, *a, **k): pass
    def header(self, *a, **k): pass
    def subheader(self, *a, **k): pass
    def write(self, *a, **k): pass
    def markdown(self, *a, **k): pass
    def caption(self, *a, **k): pass
    def text(self, *a, **k): pass
    def image(self, *a, **k): self.log.append("image")
    def info(self, m, **k): self.log.append("info: " + str(m)[:60])
    def success(self, m, **k): self.log.append("success: " + str(m)[:60])
    def warning(self, m, **k): self.log.append("warning: " + str(m)[:80])
    def error(self, m, **k): self.log.append("error: " + str(m)[:80])
    def tabs(self, labels): return [Ctx(self) for _ in labels]
    def columns(self, spec, **k): return [Ctx(self) for _ in range(spec if isinstance(spec, int) else len(spec))]
    def expander(self, *a, **k): return Ctx(self)
    def form(self, *a, **k): return Ctx(self)
    def spinner(self, *a, **k): return Ctx(self)
    @property
    def sidebar(self): return Ctx(self)
    def empty(self): return Ctx(self)
    def progress(self, v, **k): return Ctx(self)
    def rerun(self): raise RerunSignal()
    def cache_resource(self, *a, **k):
        def deco(fn):
            def w(*args):
                if args not in self._cache: self._cache[args] = fn(*args)
                return self._cache[args]
            w.__wrapped__ = fn
            return w
        return deco
    # --- widgets
    def button(self, label, key=None, on_click=None, args=(), disabled=False, **k):
        if self._hit(label) and not disabled:
            if on_click: on_click(*args)
            return True
        return False
    def form_submit_button(self, label, **k): return self._hit(label)
    def download_button(self, *a, **k): self.log.append("download_button"); return False
    def radio(self, label, options, **k): return options[self.radio_idx.get(label, 0)]
    def selectbox(self, label, options, index=0, **k): return list(options)[index] if len(list(options)) else None
    def slider(self, label, lo, hi, default, *a, **k): return default
    def checkbox(self, label, value=False, key=None, **k): return self._val(key, value)
    def text_input(self, label, value="", key=None, **k): return self._val(key, self.texts.get(label, value))
    def color_picker(self, label, value="#000000", key=None, **k): return self._val(key, value)
    def multiselect(self, label, options, default=None, key=None, **k): return self._val(key, default or [])
    def file_uploader(self, label, **k): return self.uploads.get(label, [])


def run_app(fake, ns=None):
    sys.modules["streamlit"] = types.ModuleType("streamlit")
    for n in dir(fake):
        if not n.startswith("__"):
            setattr(sys.modules["streamlit"], n, getattr(fake, n))
    sys.modules["streamlit"].session_state = fake.session_state
    ns = {"__name__": "__main__"}
    try:
        exec(compile(open(Path(__file__).resolve().parent.parent / "app.py").read(), "app.py", "exec"), ns)
    except RerunSignal:
        pass
    return ns


def main():
    import mc.config
    tmp = Path(tempfile.mkdtemp())
    mc.config.CHAR_DIR = tmp / "chars"
    mc.config.OUT_DIR = tmp / "out"
    page = T.synthetic_page()

    # A: fresh start, no pages / characters, twice (rerun)
    f = Fake(); run_app(f); run_app(f); print("A fresh start OK")

    # D: add a character through the form
    f.uploads["Coloured reference images"] = [F("r.png", encode_png(T.ref_image()))]
    f.texts["Character name"] = "Hero Girl"
    f.pressed = {"Add character"}
    run_app(f)
    f.pressed = set(); f.uploads = {}
    assert (tmp / "chars" / "Hero_Girl").exists(), "character not stored"
    print("D add character OK", [l for l in f.log if l.startswith("success")][-1:])

    # B: pages + fake models -> analyse -> quick colour -> results screen
    f.session_state["pages"] = {"p1.png": page, "p2.png": page}
    ns = run_app(f)
    eng = ns["get_engine"](False)
    eng.inject("detector", T.FakeDetector()); eng.inject("identifier", T.FakeIdent()); eng.inject("parser", T.FakeParser())
    f.pressed = {"Analyse pages"}; run_app(f)
    assert set(f.session_state["analysis"]) == {"p1.png", "p2.png"}, f.session_state["analysis"].keys()
    f.pressed = set(); f.radio_idx = {"Colouring mode": 1}
    f.session_state["analysis"]["p1.png"].panels[0].assigned = ["Hero_Girl"]
    f.pressed = {"Start colouring"}; run_app(f)
    assert set(f.session_state["results"]) == {"p1.png", "p2.png"} and f.session_state["zip"], "colouring failed"
    f.pressed = set(); run_app(f)       # results screen
    assert "download_button" in f.log and (tmp / "out" / "p1.png").exists()
    print("B analyse + colour + results OK")

    # character-card callbacks
    f.pressed = {"Auto-detect"}; run_app(f)
    assert f.session_state.get("pon_Hero_Girl_hair") is True, "auto-detect did not fill parts"
    f.pressed = {"Save"}; run_app(f)
    from mc.bank import CharacterBank
    assert "hair" in CharacterBank(tmp / "chars").chars["Hero_Girl"].parts
    f.pressed = {"Apply to all"}; f.session_state["bulk"] = ["Hero_Girl"]; run_app(f)
    f.pressed = {"Remove all pages"}; run_app(f); assert not f.session_state["pages"]
    print("callbacks OK")

    # C: AI mode with no torch installed -> clear error, no crash
    f2 = Fake(); f2.session_state["pages"] = {"p1.png": page}
    f2.pressed = {"Start colouring"}; run_app(f2)
    assert any(l.startswith("error") and "AI colouring" in l for l in f2.log), f2.log
    print("C AI-unavailable path OK:", [l for l in f2.log if l.startswith("error")][0])

    # E: delete character
    f.pressed = {"Delete"}; run_app(f)
    assert "Hero_Girl" not in CharacterBank(tmp / "chars").chars
    print("E delete OK")
    print("UI SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
