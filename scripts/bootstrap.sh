#!/usr/bin/env bash
# Prepara un equipo Ubuntu/Kubuntu para Osky Remix Engine. Idempotente.
set -euo pipefail

SURGE_VER=1.3.4
SURGE_SHA256=a6e55064487f624147d515b9ae5fc79a568b69746675b2083abde628ca7bb151

echo "== Paquetes del sistema"
sudo apt-get update -qq
sudo apt-get install -y -qq ffmpeg rubberband-cli sox fluidsynth faust csound curl

if [ ! -d "/usr/lib/vst3/Surge XT.vst3" ]; then
  echo "== Surge XT $SURGE_VER"
  tmp=$(mktemp -d)
  curl -fsSL -o "$tmp/surge.deb" \
    "https://github.com/surge-synthesizer/releases-xt/releases/download/$SURGE_VER/surge-xt-linux-x64-$SURGE_VER.deb"
  echo "$SURGE_SHA256  $tmp/surge.deb" | sha256sum -c -
  sudo apt-get install -y -qq "$tmp/surge.deb"
  rm -rf "$tmp"
fi

command -v uv >/dev/null || { echo "Instala uv: https://docs.astral.sh/uv/"; exit 1; }

echo "== Entorno Python (3.11, versiones fijadas en uv.lock)"
uv python install 3.11
uv sync

if ! command -v demucs >/dev/null; then
  echo "== Demucs (herramienta independiente)"
  uv tool install --python 3.12 demucs --with numpy --with soundfile --with torchcodec
fi

if ! command -v audio-separator >/dev/null; then
  echo "== audio-separator (RoFormer, opcional: voz y lead/backing)"
  uv tool install --python 3.12 "audio-separator[gpu]" --with audioread || echo "   (omitido: falló la instalación)"
fi

echo "== Listo. Ejecuta: uv run osky-remix doctor"
