#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
[ -d venv ] || { echo "Run ./setup.sh first"; exit 1; }
# shellcheck disable=SC1091
source venv/bin/activate
# stop Streamlit asking for an e-mail address on first launch
mkdir -p "$HOME/.streamlit"
[ -f "$HOME/.streamlit/credentials.toml" ] || printf '[general]\nemail = ""\n' > "$HOME/.streamlit/credentials.toml"
exec streamlit run app.py
