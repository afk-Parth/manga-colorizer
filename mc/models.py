"""Thin wrappers around the AI models. Everything heavy is imported lazily so the app starts instantly."""
from __future__ import annotations

import time
import zlib
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from .bank import GROUPS, Character
from .palette import apply_lines, enforce_palette  # noqa: F401


def _retry(fn: Callable, tries: int = 3, wait: float = 4.0):
    """Model downloads can hit network timeouts; retry (huggingface_hub resumes partial files)."""
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            if i < tries - 1:
                time.sleep(wait)
    raise last


def _nms(boxes: List[Tuple[float, float, float, float, float]], iou: float = 0.5):
    boxes = sorted(boxes, key=lambda b: -b[4])
    keep = []
    for b in boxes:
        ok = True
        for k in keep:
            ix = max(0, min(b[0] + b[2], k[0] + k[2]) - max(b[0], k[0]))
            iy = max(0, min(b[1] + b[3], k[1] + k[3]) - max(b[1], k[1]))
            inter = ix * iy
            if inter / (b[2] * b[3] + k[2] * k[3] - inter + 1e-6) > iou:
                ok = False
                break
        if ok:
            keep.append(b)
    return keep


# ----------------------------------------------------------------------------- person detector
class PersonDetector:
    """Zero-shot OWL-ViT: finds drawn people/characters in a panel."""

    def __init__(self, device: str):
        from transformers import pipeline
        self.pipe = _retry(lambda: pipeline("zero-shot-object-detection", model="google/owlvit-base-patch32",
                                            device=0 if device == "cuda" else -1))

    def detect(self, gray: np.ndarray, threshold: float):
        from PIL import Image
        h, w = gray.shape
        s = min(1.0, 768.0 / max(h, w))
        g = cv2.resize(gray, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else gray
        pil = Image.fromarray(np.stack([g] * 3, -1))
        res = self.pipe(pil, candidate_labels=["a drawing of a person", "a cartoon character"], threshold=threshold)
        boxes = []
        for r in res:
            b = r["box"]
            x0, y0 = max(0, b["xmin"] / s), max(0, b["ymin"] / s)
            x1, y1 = min(w, b["xmax"] / s), min(h, b["ymax"] / s)
            if x1 - x0 > 8 and y1 - y0 > 8:
                boxes.append((x0, y0, x1 - x0, y1 - y0, float(r["score"])))
        return [(int(x), int(y), int(bw), int(bh), sc) for x, y, bw, bh, sc in _nms(boxes)]


# ----------------------------------------------------------------------------- identity matching
class Identifier:
    """CLIP embeddings: which reference character does this detected person look most like?"""

    def __init__(self, device: str):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = "cuda" if device == "cuda" else "cpu"
        name = "openai/clip-vit-base-patch32"
        self.model = _retry(lambda: CLIPModel.from_pretrained(name)).to(self.device).eval()
        self.proc = _retry(lambda: CLIPProcessor.from_pretrained(name))
        self._ref_cache = {}

    def _embed(self, gray_or_rgb_list: List[np.ndarray]) -> np.ndarray:
        imgs = []
        for i in gray_or_rgb_list:
            g = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY) if i.ndim == 3 else i   # manga is grey -> compare in grey
            imgs.append(np.stack([g] * 3, -1))
        with self.torch.no_grad():
            x = self.proc(images=imgs, return_tensors="pt").to(self.device)
            e = self.model.get_image_features(**x)
            if not hasattr(e, "shape"):          # newer transformers may return an output object
                e = e.pooler_output
            e = self.torch.nn.functional.normalize(e, dim=-1)
        return e.cpu().numpy()

    def match(self, crops: List[np.ndarray], bank) -> List[Tuple[Optional[str], Optional[float]]]:
        if not bank.chars or not crops:
            return [(None, None)] * len(crops)
        for n, c in bank.chars.items():
            sig = c.signature()
            if n not in self._ref_cache or self._ref_cache[n][0] != sig:
                self._ref_cache[n] = (sig, self._embed(c.refs))
        q = self._embed(crops)
        out = []
        for v in q:
            sims = {n: float((emb @ v).max()) for n, (_, emb) in self._ref_cache.items() if n in bank.chars}
            best = max(sims, key=sims.get)
            out.append((best, sims[best]))
        return out


# ----------------------------------------------------------------------------- clothes parser
def _group_of(label: str) -> Optional[str]:
    l = label.lower()
    if "hair" in l:
        return "hair"
    if l in ("face", "left-arm", "right-arm", "left-leg", "right-leg"):
        return "skin"
    for key, g in (("upper", "top"), ("skirt", "skirt"), ("pants", "pants"), ("dress", "dress"), ("shoe", "shoes"),
                   ("bag", "bag"), ("hat", "hat"), ("scarf", "scarf"), ("belt", "belt")):
        if key in l:
            return g
    return None


class ClothesParser:
    """SegFormer-B2 (ATR human parsing): hair / face+skin / top / skirt / pants / dress / shoes / bag ..."""

    def __init__(self, device: str):
        import torch
        from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor
        self.torch = torch
        self.device = "cuda" if device == "cuda" else "cpu"
        name = "mattmdjaga/segformer_b2_clothes"
        self.proc = _retry(lambda: SegformerImageProcessor.from_pretrained(name))
        self.model = _retry(lambda: AutoModelForSemanticSegmentation.from_pretrained(name)).to(self.device).eval()
        lut = np.zeros(max(int(k) for k in self.model.config.id2label) + 1, np.uint8)
        for k, v in self.model.config.id2label.items():
            g = _group_of(v)
            if g in GROUPS:
                lut[int(k)] = GROUPS.index(g) + 1
        self.lut = lut

    def parse(self, rgb: np.ndarray) -> np.ndarray:
        """Returns uint8 map (H,W): 0 = none, i+1 = GROUPS[i]."""
        from PIL import Image
        h, w = rgb.shape[:2]
        inputs = self.proc(images=Image.fromarray(rgb), return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
            up = self.torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
            lab = up.argmax(dim=1)[0].cpu().numpy()
        return self.lut[lab]


# ----------------------------------------------------------------------------- colourisers
def make_reference(chars: List[Character]) -> np.ndarray:
    """Square grid of up to 4 character reference images -> single IP-Adapter reference image."""
    tiles = [c.refs[0] for c in chars[:4]]
    S = 448
    canvas = np.full((S, S, 3), 255, np.uint8)
    n = len(tiles)
    cols = 1 if n == 1 else 2
    rows = (n + cols - 1) // cols
    tw, th = S // cols, S // rows
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        s = min(tw / t.shape[1], th / t.shape[0])
        t2 = cv2.resize(t, (max(1, int(t.shape[1] * s)), max(1, int(t.shape[0] * s))), interpolation=cv2.INTER_AREA)
        y0 = r * th + (th - t2.shape[0]) // 2
        x0 = c * tw + (tw - t2.shape[1]) // 2
        canvas[y0:y0 + t2.shape[0], x0:x0 + t2.shape[1]] = t2
    return canvas


class QuickColorizer:
    """No-AI preview: fills the regions between lines with the character's palette. Instant, CPU only."""

    def __call__(self, gray: np.ndarray, chars: List[Character], st) -> np.ndarray:
        pal = chars[0].palette if chars and chars[0].palette else [(200, 170, 140), (90, 120, 170), (60, 50, 50)]
        pal = np.array(pal, np.uint8)
        pal = pal[np.argsort(pal.astype(int).sum(1))]
        g = cv2.GaussianBlur(gray, (0, 0), 2)
        _, free = cv2.threshold(g, 150, 255, cv2.THRESH_BINARY)
        n, lab = cv2.connectedComponents(free, connectivity=4)
        out = np.full(gray.shape + (3,), 255, np.uint8)
        for i in range(1, n):
            m = lab == i
            if m.sum() > 0.4 * gray.size and gray[m].mean() > 235:
                continue   # big white area = sky/background
            level = 1 - gray[m].mean() / 255.0
            out[m] = pal[min(int(level * len(pal) * 1.5), len(pal) - 1)]
        return out


class DiffusionColorizer:
    """Stable Diffusion 1.5 + anime line-art ControlNet + IP-Adapter (character reference image)."""

    BASE = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    CN = "lllyasviel/control_v11p_sd15s2_lineart_anime"

    def __init__(self, device: str, low_memory: bool = False):
        self.device, self.low_memory = device, low_memory
        self.dtype_name = "float16" if device in ("cuda", "mps") else "float32"
        self.pipe = None
        _retry(lambda: self._load(self.dtype_name))

    def _load(self, dtype_name: str):
        import torch
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
        self.torch = torch
        dtype = getattr(torch, dtype_name)
        self.pipe = None
        variant = "fp16" if dtype_name == "float16" else None
        cn = ControlNetModel.from_pretrained(
            self.CN, torch_dtype=dtype, variant=variant, use_safetensors=True,
        )
        kw = dict(controlnet=cn, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False,
                  use_safetensors=True)
        if variant:
            kw["variant"] = variant
        pipe = StableDiffusionControlNetPipeline.from_pretrained(self.BASE, **kw)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
        if self.device == "cuda" and self.low_memory:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)
        if self.low_memory or self.device in ("mps", "cpu"):
            pipe.enable_attention_slicing()
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass
        self.pipe, self.dtype_name = pipe, dtype_name

    def _run(self, gray, chars, st):
        from PIL import Image
        h, w = gray.shape
        s = st.size / max(h, w)
        nw, nh = max(64, int(round(w * s / 8)) * 8), max(64, int(round(h * s / 8)) * 8)
        g = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
        # screentone-tolerant line extraction; solid blacks (hair, shadows) stay solid
        lines = cv2.adaptiveThreshold(cv2.medianBlur(g, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
        lines = np.where(g < 80, 0, lines).astype(np.uint8)
        ctrl = Image.fromarray(np.stack([255 - lines] * 3, -1))   # ControlNet wants white lines on black

        prompt = "masterpiece, best quality, full color anime illustration, flat colors, vibrant, clean lineart"
        if len(chars) > 1:
            prompt = f"{len(chars)} people, " + prompt
        extra = [c.prompt() for c in chars if c.prompt()]
        if extra:
            prompt += ", " + ", ".join(extra)
        if chars:
            ref = Image.fromarray(make_reference(chars))
            self.pipe.set_ip_adapter_scale(st.ip_scale)
            seed = zlib.crc32("+".join(sorted(c.name for c in chars)).encode())
        else:
            ref = Image.new("RGB", (224, 224), (128, 128, 128))
            self.pipe.set_ip_adapter_scale(0.0)
            seed = 1234
        gen = self.torch.Generator("cpu").manual_seed(seed)
        res = self.pipe(prompt=prompt, image=ctrl, ip_adapter_image=ref, width=nw, height=nh,
                        num_inference_steps=st.steps, guidance_scale=st.guidance,
                        controlnet_conditioning_scale=st.control_scale,
                        negative_prompt="monochrome, grayscale, lowres, bad anatomy, text, watermark, blurry",
                        generator=gen).images[0]
        return np.array(res.convert("RGB"))

    def __call__(self, gray: np.ndarray, chars: List[Character], st) -> np.ndarray:
        arr = self._run(gray, chars, st)
        if arr.max() < 8 and self.dtype_name != "float32":    # fp16 'black image' bug on some GPUs -> redo in fp32
            self._load("float32")
            arr = self._run(gray, chars, st)
        h, w = gray.shape
        return cv2.resize(arr, (w, h), interpolation=cv2.INTER_CUBIC)
