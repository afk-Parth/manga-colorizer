#!/usr/bin/env bash
# Optional: download all AI models now (~6 GB) instead of at first use. Re-run if it times out; it resumes.
cd "$(dirname "$0")" || exit 1
source venv/bin/activate
python -m mc doctor --models
