"""Lazy model holder. A model that fails to load is reported (engine.errors) and skipped - never a crash."""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

from .config import pick_device


class Engine:
    def __init__(self, low_memory: bool = False):
        self.device = pick_device()
        self.aux_device = "cuda" if self.device == "cuda" else "cpu"   # small helper models run on CPU for stability
        self.low_memory = low_memory
        self.errors: Dict[str, str] = {}
        self._obj: Dict[str, object] = {}
        self._failed = set()

    # --- dependency check ---------------------------------------------------------------
    @staticmethod
    def ai_installed() -> Tuple[bool, str]:
        missing = []
        for m in ("torch", "transformers", "diffusers"):
            try:
                __import__(m)
            except Exception:
                missing.append(m)
        if missing:
            return False, "Missing packages: " + ", ".join(missing) + ".  Run ./setup.sh again (or: pip install -r requirements-ai.txt)"
        return True, ""

    def inject(self, key: str, obj):
        self._obj[key] = obj
        self._failed.discard(key)

    def retry_failed(self):
        self._failed.clear()

    def model_loaded(self, key: str) -> bool:
        return key in self._obj

    def _get(self, key: str, factory: Callable[[], object]):
        if key in self._obj:
            return self._obj[key]
        if key in self._failed:
            return None
        ok, msg = self.ai_installed()
        if not ok:
            self.errors[key] = msg
            self._failed.add(key)
            return None
        try:
            obj = factory()
        except Exception as e:  # noqa: BLE001
            self.errors[key] = f"{type(e).__name__}: {e}"
            self._failed.add(key)
            return None
        self._obj[key] = obj
        self.errors.pop(key, None)
        return obj

    @property
    def detector(self):
        from .models import PersonDetector
        return self._get("detector", lambda: PersonDetector(self.aux_device))

    @property
    def identifier(self):
        from .models import Identifier
        return self._get("identifier", lambda: Identifier(self.aux_device))

    @property
    def parser(self):
        from .models import ClothesParser
        return self._get("parser", lambda: ClothesParser(self.aux_device))

    def get_colorizer(self, backend: str):
        """-> (colorizer | None, error message | None)"""
        from .models import DiffusionColorizer, QuickColorizer
        if backend == "quick":
            return QuickColorizer(), None
        c = self._get("diffusion", lambda: DiffusionColorizer(self.device, self.low_memory))
        return c, (None if c is not None else self.errors.get("diffusion", "unknown error"))
