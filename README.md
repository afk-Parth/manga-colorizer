# 🎨 Manga Colorizer

Black & white manga in → coloured manga out, with the **same colours for each character on every page**.
Runs locally in your browser (Streamlit). Works on Apple-Silicon Macs (uses the GPU), NVIDIA GPUs, or CPU (slow).

## 1. Install (once)
```bash
cd manga-colorizer
bash setup.sh
```
If it stops with a network timeout, just run `bash setup.sh` again. Python 3.9+ is required, 3.11 recommended (`brew install python@3.11`).

## 2. Start
```bash
bash run_ui.sh
```
Your browser opens at http://localhost:8501.

## 3. Use it (3 tabs)
1. **Characters** - add each main character with 1-4 *coloured* pictures of them. Press **Auto-detect from pictures** to read their hair / skin / top / skirt / shoes… colours (fix any colour with the pickers, tick the parts you want locked, Save).
2. **Pages & layout** - upload pages (images, PDF, CBZ) → **Analyse pages**. The app finds panels, speech bubbles and characters and draws them for you. Fix "who is in each panel" with the dropdowns if needed.
3. **Colourise & download** - press **Start colouring**. Download each page or a zip.

**Tip:** first run with *Quick preview* (sidebar) - instant, no AI - to check panel/character detection. Then switch to *AI colouring*.
The first AI run downloads ~6 GB of models (one time). `bash download_models.sh` does that in advance.

## How the colours stay consistent
| Step | What it does |
|---|---|
| Character bank | Stores reference pictures + a colour per body part (hair, skin, top, skirt, pants, dress, shoes, bag…) |
| Reference conditioning | The character's picture steers the AI (IP-Adapter) and a fixed per-character seed is used |
| Prompt | Colours are turned into words ("black hair, white top, red skirt") |
| Colour correction | After colouring, a clothes-segmentation model finds hair/skin/clothes in each character and shifts them to the stored colours |
| Bubbles | Speech bubbles and lettering are left untouched |

Models used (all downloaded automatically): Stable Diffusion 1.5 + anime line-art ControlNet + IP-Adapter (colouring), OWL-ViT (finds characters), CLIP (names them), SegFormer-B2 clothes (finds hair/clothes). No model is trained by this project.

## No UI? Command line
```bash
source venv/bin/activate
python -m mc doctor                 # check installation
python -m mc doctor --models        # download + test every model
python -m mc run pages/ -o out --quick     # no AI
python -m mc run pages/ -o out             # AI
```

## Troubleshooting
- **Timeouts during install/download** → run the same command again; finished parts are kept.
- **"Out of memory" / very slow** → sidebar: Quality = Fast, tick *Low-memory mode*.
- **A panel stays black & white** → the page shows which panel failed and why; the rest is still coloured.
- **Wrong character in a panel** → fix it in tab 2 (dropdown under the page preview).
- **Colours still drift** → raise *Colour-consistency correction*, make sure the part colours are ticked/saved, use better reference pictures.

## Honest limitations
- Character *naming* uses CLIP similarity - the weakest link. That's why tab 2 lets you correct it in two clicks. Use "Set every panel to…" for single-protagonist chapters.
- The clothes model was trained on photos of people; it works best on the *coloured output* and on reasonably realistic anime art. On unusual styles it may find nothing → the app then falls back to palette matching.
- Panel detection is classic image processing: normal layouts are fine, borderless/overlapping layouts may merge panels.
- Quality is that of Stable Diffusion 1.5 + generic anime ControlNet; heavy screentone can confuse the line extraction.
