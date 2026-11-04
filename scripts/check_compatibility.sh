#!/usr/bin/env bash
# Ejecuta el doctor y las pruebas de compatibilidad. Sale con error si falla alguna crítica.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run osky-remix doctor "$@"
uv run pytest -q -m compat
