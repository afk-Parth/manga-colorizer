"""Manga Colorizer - Streamlit UI.   Start with:  ./run_ui.sh"""
from __future__ import annotations

import io
import hmac
import os
import time
import uuid
import zipfile
from pathlib import Path

import streamlit as st

from mc.bank import GROUPS, CharacterBank, safe_name
from mc.config import CHAR_DIR, OUT_DIR, Settings
from mc.consistency import detect_parts
from mc.engine import Engine
from mc.io_utils import natural_key, pages_from_bytes
from mc.palette import hex_to_rgb, rgb_to_hex
from mc.pipeline import analyse_page, colorize_page, prepare_characters
from mc.io_utils import encode_png
from mc.viz import draw_overlay

st.set_page_config(page_title="Manga Colorizer", page_icon="🎨", layout="wide")

PRESETS = {"Fast (512 px, 15 steps)": (512, 15), "Balanced (640 px, 20 steps)": (640, 20), "High (768 px, 28 steps)": (768, 28)}
IMG_TYPES = ["png", "jpg", "jpeg", "webp", "bmp", "avif"]
ss = st.session_state


def app_setting(name: str, default: str = "") -> str:
    if name in os.environ:
        return os.environ[name]
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


for _k, _v in (("pages", {}), ("seen", set()), ("analysis", {}), ("results", {}), ("problems", {}), ("up_gen", 0), ("zip", None)):
    if _k not in ss:
        ss[_k] = _v
AI_READY, AI_MISSING = Engine.ai_installed()
QUICK_ONLY = app_setting("MANGA_COLORIZER_QUICK_ONLY").lower() in {"1", "true", "yes"} or not AI_READY
ISOLATE_SESSIONS = app_setting("MANGA_COLORIZER_ISOLATE_SESSIONS").lower() in {"1", "true", "yes"}
if ISOLATE_SESSIONS:
    ss.setdefault("storage_session_id", uuid.uuid4().hex)
CHARACTER_DIR = CHAR_DIR / ss["storage_session_id"] if ISOLATE_SESSIONS else CHAR_DIR
SESSION_OUTPUT_DIR = OUT_DIR / ss["storage_session_id"] if ISOLATE_SESSIONS else OUT_DIR
ACCESS_PASSWORD = app_setting("MANGA_COLORIZER_ACCESS_PASSWORD")
PASSWORD_REQUIRED = app_setting("MANGA_COLORIZER_REQUIRE_PASSWORD").lower() in {"1", "true", "yes"}
if PASSWORD_REQUIRED and not ACCESS_PASSWORD:
    st.error("Server setup required: configure MANGA_COLORIZER_ACCESS_PASSWORD in the deployment environment.")
    st.stop()
if ACCESS_PASSWORD and not ss.get("authenticated", False):
    st.title("Manga Colorizer")
    with st.form("server_access"):
        entered_password = st.text_input("Server access password", type="password")
        submitted_password = st.form_submit_button("Continue")
    if submitted_password:
        if hmac.compare_digest(entered_password, ACCESS_PASSWORD):
            ss["authenticated"] = True
            st.rerun()
        st.error("Incorrect password.")
    st.stop()


@st.cache_resource(show_spinner=False)
def get_engine(low_memory: bool) -> Engine:
    return Engine(low_memory=low_memory)


def swatch_html(colors) -> str:
    return "".join(
        f"<span style='display:inline-block;width:30px;height:30px;border-radius:6px;margin:0 5px 5px 0;"
        f"background:{rgb_to_hex(c)};border:1px solid #8888'></span>" for c in colors)


# ------------------------------------------------------------------ sidebar / settings
with st.sidebar:
    st.header("⚙️ Settings")
    if QUICK_ONLY:
        mode = "Quick preview (instant, flat colours)"
        st.caption("Quick preview only on this host.")
    else:
        mode = st.radio("Colouring mode", ["AI colouring (best quality)", "Quick preview (instant, flat colours)"])
    preset = st.selectbox("Quality", list(PRESETS), index=1)
    rtl = st.radio("Reading direction", ["Right-to-left (manga)", "Left-to-right"]).startswith("Right")
    with st.expander("Advanced"):
        ip_scale = st.slider("Follow character reference", 0.3, 1.0, 0.8, 0.05)
        control_scale = st.slider("Follow line art", 0.5, 1.5, 1.0, 0.05)
        enforce_s = st.slider("Colour-consistency correction", 0.0, 1.0, 0.6, 0.05)
        detect_chars = st.checkbox("Detect characters automatically", value=True)
        use_parts = st.checkbox("Correct hair / skin / clothes colours", value=True)
        det_thr = st.slider("Character detection sensitivity (lower finds more)", 0.02, 0.4, 0.08, 0.01)
        clip_thr = st.slider("Name-matching strictness", 0.4, 0.95, 0.65, 0.01)
        low_mem = st.checkbox("Low-memory mode (slower)", value=False)

S = Settings(backend="quick" if mode.startswith("Quick") else "ai", size=PRESETS[preset][0], steps=PRESETS[preset][1],
             ip_scale=ip_scale, control_scale=control_scale, enforce=enforce_s, rtl=rtl, detect_characters=detect_chars,
             use_parts=use_parts, det_threshold=det_thr, clip_threshold=clip_thr, low_memory=low_mem)
engine = get_engine(low_mem)
bank = CharacterBank(CHARACTER_DIR)

with st.sidebar:
    st.caption(f"Compute device: **{engine.device}**")
    ok, why = engine.ai_installed()
    if not ok:
        st.error(why)
    for _key, _msg in list(engine.errors.items()):
        st.warning(f"{_key}: {_msg}")


# ------------------------------------------------------------------ callbacks (run before the page re-renders)
def cb_save(name):
    ch = CharacterBank(CHARACTER_DIR).chars.get(name)
    if ch is None:
        return
    ch.outfit = str(ss.get(f"outfit_{name}", "")).strip()
    ch.parts = {g: hex_to_rgb(ss[f"pcol_{name}_{g}"]) for g in GROUPS if ss.get(f"pon_{name}_{g}")}
    ch.save_meta()
    ss["_msg"] = ("success", f"Saved {name}.")


def cb_detect(name, low_memory):
    eng = get_engine(low_memory)
    eng.retry_failed()
    ch = CharacterBank(CHARACTER_DIR).chars.get(name)
    parser = eng.parser
    if ch is None:
        return
    if parser is None:
        ss["_msg"] = ("error", "Clothes model could not be loaded: " + eng.errors.get("parser", "unknown error"))
        return
    parts = detect_parts(ch, parser)
    if not parts:
        ss["_msg"] = ("warning", "Nothing detected in the references. Use clear full-colour pictures of the character, or set the colours manually.")
        return
    ch.parts = parts
    ch.save_meta()
    for g in GROUPS:
        ss[f"pon_{name}_{g}"] = g in parts
        ss[f"pcol_{name}_{g}"] = rgb_to_hex(parts[g]) if g in parts else "#808080"
    ss["_msg"] = ("success", "Detected: " + ", ".join(parts) + ". Adjust any colour below, then press Save.")


def cb_delete(name):
    CharacterBank(CHARACTER_DIR).delete(name)
    for k in [k for k in list(ss.keys()) if isinstance(k, str) and (k.startswith(f"pon_{name}_") or k.startswith(f"pcol_{name}_") or k == f"outfit_{name}")]:
        del ss[k]
    ss["_msg"] = ("info", f"Deleted {name}.")


def cb_clear_pages():
    ss["pages"], ss["seen"], ss["analysis"], ss["results"], ss["problems"], ss["zip"] = {}, set(), {}, {}, {}, None
    ss["up_gen"] += 1


def cb_apply_all():
    sel = list(ss.get("bulk", []))
    for pg, an in ss["analysis"].items():
        for i, p in enumerate(an.panels):
            p.assigned = list(sel)
            ss[f"as_{pg}_{i}"] = list(sel)


# ------------------------------------------------------------------ header
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono:wght@400;500&family=Noto+Sans+JP:wght@400;500;600;700;900&display=swap');
:root {
    --paper: #f4f2ed;
    --ink: #171717;
    --muted: #66635f;
    --line: #c9c5be;
    --moss: #252525;
    --coral: #d83227;
    --sidebar: #171717;
}
.stApp { background: var(--paper); color: var(--ink); font-family: 'Noto Sans JP', sans-serif; }
[data-testid="stAppViewContainer"] > .main { background-image: radial-gradient(#17171713 .7px, transparent .8px); background-size: 7px 7px; }
[data-testid="stMainBlockContainer"] { max-width: 1480px; padding-top: 2rem; }
section[data-testid="stSidebar"] { background: var(--sidebar); border-right: 2px solid var(--coral); }
section[data-testid="stSidebar"] * { color: #f3f2ec; }
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: #c3cec5; }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #c3cec5; }
section[data-testid="stSidebar"] [data-testid="stExpander"] { border-color: #4b4b4b; }
.studio-head { position: relative; isolation: isolate; overflow: hidden; display: flex; align-items: flex-end; justify-content: space-between; gap: 1.5rem; padding: 1.6rem 1.8rem 1.45rem; margin-bottom: 1.2rem; border: 2px solid var(--ink); background: var(--paper); box-shadow: 5px 5px 0 var(--ink); }
.studio-head::before { position: absolute; z-index: -1; content: ""; inset: 0 0 0 57%; background: radial-gradient(#17171728 1px, transparent 1.4px); background-size: 8px 8px; }
.studio-head::after { position: absolute; z-index: -1; content: ""; top: 0; right: 0; width: 9px; height: 100%; background: var(--coral); }
.studio-kicker { margin: 0 0 .2rem; color: var(--coral); font: 500 11px 'DM Mono', monospace; letter-spacing: 0; text-transform: uppercase; }
.studio-title { margin: 0; color: var(--ink); font: 400 48px/.96 'Bebas Neue', 'Noto Sans JP', sans-serif; letter-spacing: 0; text-transform: uppercase; }
.studio-note { margin: .55rem 0 0; color: var(--muted); font-size: 14px; }
.studio-mode { flex: 0 0 auto; padding: .6rem .75rem; border: 1px solid var(--ink); background: var(--paper); color: var(--ink); font: 500 11px 'DM Mono', monospace; text-align: right; text-transform: uppercase; }
.studio-art { flex: 0 1 360px; min-width: 250px; display: grid; grid-template-columns: 1.2fr .8fr 1fr; gap: 5px; height: 122px; transform: skew(-3deg); }
.manga-frame { position: relative; display: flex; align-items: center; justify-content: center; overflow: hidden; border: 2px solid var(--ink); background-color: var(--paper); }
.manga-frame b { position: relative; z-index: 1; color: var(--ink); font: 900 47px/1 'Noto Sans JP', sans-serif; }
.manga-frame small { position: absolute; left: 6px; bottom: 5px; color: var(--ink); font: 500 8px 'DM Mono', monospace; }
.manga-frame--ink { background-color: #191919; background-image: repeating-linear-gradient(132deg, transparent 0 8px, #ffffff22 9px 10px); }
.manga-frame--ink b, .manga-frame--ink small { color: #f4f2ed; }
.manga-frame--tone { background-image: radial-gradient(#171717b8 .9px, transparent 1.2px); background-size: 6px 6px; }
.manga-frame--red { background: var(--coral); }
.manga-frame--red b { color: #fff; font-size: 32px; }
.studio-art .studio-mode { position: absolute; right: -1px; bottom: -1px; z-index: 2; padding: .35rem .5rem; font-size: 9px; }
.stTabs [data-baseweb="tab-list"] { gap: 1.3rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { height: 48px; padding: 0 .15rem; color: var(--muted); font-weight: 600; }
.stTabs [aria-selected="true"] { color: var(--ink) !important; border-bottom-color: var(--coral) !important; }
.stButton > button, .stDownloadButton > button { border-radius: 2px; border: 1px solid var(--ink); color: var(--ink); background: #fbfaf6; font-weight: 700; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--moss); color: var(--moss); }
.stButton > button[kind="primary"] { background: var(--coral); border-color: var(--coral); color: white; }
.stButton > button[kind="primary"]:hover { background: #b5251c; border-color: #b5251c; color: white; }
.stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"], .stMultiSelect [data-baseweb="select"] { border-radius: 4px; }
[data-testid="stFileUploader"] section { border: 1px dashed #77716b; border-radius: 2px; background: #faf9f4; }
[data-testid="stExpander"] { border-color: var(--line); border-radius: 2px; background: #faf9f5; }
[data-testid="stProgress"] > div > div { background: var(--moss); }
@media (max-width: 700px) {
    [data-testid="stMainBlockContainer"] { padding: 1rem .85rem; }
    .studio-head { align-items: flex-start; flex-direction: column; gap: .65rem; padding: 1.2rem; }
    .studio-title { font-size: 40px; }
    .studio-mode { text-align: left; }
    .studio-art { flex: 0 0 auto; width: 100%; max-width: 360px; min-width: 0; height: 88px; }
    .manga-frame b { font-size: 36px; }
    .manga-frame--red b { font-size: 25px; }
    .stTabs [data-baseweb="tab-list"] { gap: .55rem; }
    .stTabs [data-baseweb="tab"] { font-size: 12px; padding: 0 .1rem; }
}
</style>
""", unsafe_allow_html=True)
st.markdown(
        f"""<header class="studio-head">
            <div><p class="studio-kicker">Manga Color Lab / Local Studio</p>
            <h1 class="studio-title">Manga Colorizer</h1>
            <p class="studio-note">Build a character palette, map the page, then color panel by panel.</p></div>
            <div class="studio-art" aria-hidden="true">
                <div class="manga-frame manga-frame--ink"><b>彩</b><small>INK / 01</small></div>
                <div class="manga-frame manga-frame--tone"><b>影</b></div>
                <div class="manga-frame manga-frame--red"><b>漫画</b><small>COLOR LAB</small></div>
                <div class="studio-mode">{mode.split(' (')[0]}<br>{engine.device} compute</div>
            </div>
        </header>""",
        unsafe_allow_html=True,
)
_msg = ss.pop("_msg", None)
if _msg:
    getattr(st, _msg[0])(_msg[1])

tab_c, tab_p, tab_r = st.tabs(["1 · Characters", "2 · Pages & layout", "3 · Colourise & download"])

# ------------------------------------------------------------------ TAB 1: characters
with tab_c:
    st.subheader("Characters")
    st.write("Add each main character once with **1-4 coloured reference images** (any colour picture of them: official art, cover, a coloured page). "
             "The app remembers their look and keeps hair, skin and clothes colours the same on every page.")
    with st.form("add_char", clear_on_submit=True):
        c_name = st.text_input("Character name")
        c_files = st.file_uploader("Coloured reference images", type=IMG_TYPES, accept_multiple_files=True)
        c_outfit = st.text_input("Extra description (optional)", placeholder="e.g. short black hair, school uniform, red ribbon")
        submitted = st.form_submit_button("➕ Add character")
    added = False
    if submitted:
        try:
            if not c_files:
                raise ValueError("Please upload at least one coloured reference image.")
            bank.add(c_name, [(f.name, f.getvalue()) for f in c_files], c_outfit)
            ss["_msg"] = ("success", f"Added {safe_name(c_name)}. Press 'Auto-detect' on its card to read hair / clothes colours from the pictures.")
            added = True
        except ValueError as e:
            st.error(str(e))
    if added:
        st.rerun()

    if not bank.chars:
        st.info("No characters yet. You can still colour without them, but colours may differ between pages.")
    for name in bank.names():
        ch = bank.chars[name]
        with st.expander(f"👤 {name}", expanded=(len(bank.chars) == 1)):
            cols = st.columns(max(1, min(4, len(ch.refs))))
            for i, r in enumerate(ch.refs[:4]):
                cols[i].image(r, width=130)
            st.caption("Dominant colours found in the references:")
            st.markdown(swatch_html(ch.palette), unsafe_allow_html=True)
            if f"outfit_{name}" not in ss:
                ss[f"outfit_{name}"] = ch.outfit
            st.text_input("Extra description", key=f"outfit_{name}")
            st.markdown("**Hair / skin / clothes colours** - tick a part to lock its colour on every page:")
            pcols = st.columns(4)
            for gi, g in enumerate(GROUPS):
                cur = ch.parts.get(g)
                kon, kcol = f"pon_{name}_{g}", f"pcol_{name}_{g}"
                if kon not in ss:
                    ss[kon] = cur is not None
                if kcol not in ss:
                    ss[kcol] = rgb_to_hex(cur) if cur else "#808080"
                with pcols[gi % 4]:
                    st.checkbox(g, key=kon)
                    st.color_picker(f"{g} colour", key=kcol, label_visibility="collapsed")
            b1, b2, b3 = st.columns(3)
            b1.button("💾 Save", key=f"save_{name}", on_click=cb_save, args=(name,))
            b2.button("🔍 Auto-detect from pictures", key=f"det_{name}", on_click=cb_detect, args=(name, low_mem))
            b3.button("🗑 Delete", key=f"del_{name}", on_click=cb_delete, args=(name,))

# ------------------------------------------------------------------ TAB 2: pages
with tab_p:
    st.subheader("Pages")
    ups = st.file_uploader("Upload black & white manga pages (images, PDF or CBZ)", type=IMG_TYPES + ["pdf", "cbz", "zip"],
                           accept_multiple_files=True, key=f"page_up_{ss['up_gen']}")
    for f in sorted([f for f in (ups or []) if f"{f.name}:{f.size}" not in ss["seen"]], key=lambda f: natural_key(f.name)):
        try:
            for pname, g in pages_from_bytes(f.name, f.getvalue()):
                base, k = Path(pname).stem, 2
                while pname in ss["pages"]:
                    pname, k = f"{base}_{k}.png", k + 1
                ss["pages"][pname] = g
            ss["seen"].add(f"{f.name}:{f.size}")
        except Exception as e:  # noqa: BLE001
            st.error(f"{f.name}: {e}")

    names = bank.names()
    if "always" not in ss:
        ss["always"] = names if len(names) == 1 else []
    ss["always"] = [n for n in ss["always"] if n in names]
    always = st.multiselect("Characters who appear in (almost) every panel - used when detection can't find them", names, key="always")

    c1, c2 = st.columns(2)
    run_analysis = c1.button("🔎 Analyse pages (find panels & characters)", disabled=not ss["pages"])
    c2.button("🧹 Remove all pages", on_click=cb_clear_pages, disabled=not ss["pages"])

    if run_analysis:
        with st.spinner("Loading detection models... (the first time this downloads about 1 GB)"):
            if S.detect_characters:
                _ = engine.detector
                if bank.chars:
                    _ = engine.identifier
        bar = st.progress(0.0)
        for i, (pname, g) in enumerate(list(ss["pages"].items())):
            ss["analysis"][pname] = analyse_page(pname, g, bank, engine, S, always)
            for k in [k for k in list(ss.keys()) if isinstance(k, str) and k.startswith(f"as_{pname}_")]:
                del ss[k]
            bar.progress((i + 1) / len(ss["pages"]))
        for _key, _m in engine.errors.items():
            st.warning(f"{_key}: {_m}  -> panels are still found; assign characters by hand below.")

    if not ss["pages"]:
        st.info("Upload some pages to begin.")
    else:
        st.caption(f"{len(ss['pages'])} page(s) loaded.")
        page = st.selectbox("Review page", list(ss["pages"]))
        gray, an = ss["pages"][page], ss["analysis"].get(page)
        if an is None:
            st.image(gray, caption="Not analysed yet - press 'Analyse pages' above.")
        else:
            left, right = st.columns([3, 2])
            left.image(draw_overlay(gray, an), caption="Blue = panels (number: characters assigned) · green box = recognised character · "
                                                       "orange box = detected but unnamed · green tint = speech bubble (left untouched)")
            with right:
                st.markdown("**Who is in each panel?**  (fix anything wrong - this is what the colouring uses)")
                for i, p in enumerate(an.panels):
                    key = f"as_{page}_{i}"
                    if key not in ss:
                        ss[key] = [n for n in p.assigned if n in names]
                    ss[key] = [n for n in ss[key] if n in names]
                    sel = st.multiselect(f"Panel {i + 1}", names, key=key)
                    p.assigned = list(sel)
                st.multiselect("Set every panel on every page to:", names, key="bulk")
                st.button("Apply to all panels", on_click=cb_apply_all)

# ------------------------------------------------------------------ TAB 3: colourise
def do_colour(targets):
    with st.spinner("Loading models... the first AI run downloads about 6 GB, keep this window open."):
        colorizer, err = engine.get_colorizer(S.backend)
    if colorizer is None:
        st.error(f"Could not start AI colouring:\n\n{err}")
        st.info("Fix the problem above (usually a network timeout - just press Start again) or choose 'Quick preview' in the sidebar.")
        engine.retry_failed()
        return
    with st.spinner("Preparing characters..."):
        for m in prepare_characters(bank, engine, S):
            st.warning(m)
    bar, status, t0 = st.progress(0.0), st.empty(), time.time()
    for pi, name in enumerate(targets):
        gray = ss["pages"][name]
        an = ss["analysis"].get(name)
        if an is None:
            status.text(f"Analysing {name} ...")
            an = analyse_page(name, gray, bank, engine, S, ss.get("always", []))
            ss["analysis"][name] = an

        def cb(i, n, text, pi=pi, name=name):
            bar.progress(min(1.0, (pi + i / max(n, 1)) / len(targets)))
            status.text(f"{name} · {text} · {int(time.time() - t0)}s elapsed")

        rgb, probs = colorize_page(gray, an, bank, engine, S, colorizer, cb)
        png = encode_png(rgb)
        ss["results"][name] = png
        SESSION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (SESSION_OUTPUT_DIR / name).write_bytes(png)
        if probs:
            ss["problems"][name] = probs
        else:
            ss["problems"].pop(name, None)
    bar.progress(1.0)
    status.text(f"Done in {int(time.time() - t0)}s. Files are also saved in {SESSION_OUTPUT_DIR}")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for n in sorted(ss["results"], key=natural_key):
            z.writestr(n, ss["results"][n])
    ss["zip"] = buf.getvalue()


with tab_r:
    st.subheader("Colourise")
    if not ss["pages"]:
        st.info("Upload pages in tab 2 first.")
    else:
        if S.backend == "ai":
            st.caption(f"AI mode on **{engine.device}**. Expect roughly 20-60 s per panel on a recent Mac GPU, much longer on CPU. "
                       "Try 'Quick preview' first to check the layout.")
        chosen = st.multiselect("Pages to colourise (leave empty = all pages)", list(ss["pages"]))
        if st.button("🎨 Start colouring", type="primary"):
            do_colour(chosen or list(ss["pages"]))

    for pname in sorted(ss["results"], key=natural_key):
        with st.expander(f"📄 {pname}", expanded=True):
            if ss["problems"].get(pname):
                st.warning("Some panels could not be coloured and were left black & white:\n\n" + "\n".join(f"- {x}" for x in ss["problems"][pname]))
            a, b = st.columns(2)
            if pname in ss["pages"]:
                a.image(ss["pages"][pname], caption="Original")
            b.image(ss["results"][pname], caption="Coloured")
            st.download_button("⬇️ Download PNG", data=ss["results"][pname], file_name=pname, mime="image/png", key=f"dl_{pname}")
    if ss["results"] and ss["zip"]:
        st.download_button("⬇️ Download all pages (.zip)", data=ss["zip"], file_name="coloured_pages.zip", mime="application/zip", key="dl_zip")
