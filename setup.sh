#!/usr/bin/env bash
# One-time setup. Safe to run again if anything fails (e.g. a network timeout) - it resumes where it can.
cd "$(dirname "$0")" || exit 1

PY=""
for c in python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)'; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "Python 3.9+ not found. Install it with:  brew install python@3.11   (then run this again)"; exit 1
fi
echo "Using $($PY --version) ($PY)"
if "$PY" -c 'import sys; sys.exit(0 if sys.version_info < (3,10) else 1)'; then
  echo "NOTE: Python 3.9 works but is old. For best results:  brew install python@3.11  and re-run."
fi

[ -d venv ] || "$PY" -m venv venv || { echo "Could not create venv"; exit 1; }
# shellcheck disable=SC1091
source venv/bin/activate
export PIP_DEFAULT_TIMEOUT=120

retry() { n=0; until "$@"; do n=$((n+1)); [ $n -ge 8 ] && return 1; echo "  ...network hiccup, retrying ($n/8)"; sleep 4; done; }

echo "== Upgrading pip";                 retry python -m pip install --upgrade pip
echo "== Installing app (small)";        retry python -m pip install -r requirements-base.txt || { echo "Base install failed - run ./setup.sh again"; exit 1; }
echo "== Installing AI libraries (big, a few minutes)"
retry python -m pip install -r requirements-ai.txt   || echo "AI install failed - run ./setup.sh again. (Quick-preview mode already works.)"
echo "== Installing PDF support (optional)"; python -m pip install -r requirements-pdf.txt >/dev/null 2>&1 || echo "  (PDF support skipped - images and CBZ still work)"

echo; python -m mc doctor
echo; echo "Setup finished.  Start the app with:  ./run_ui.sh"
