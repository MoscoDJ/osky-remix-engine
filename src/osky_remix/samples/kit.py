"""Kits de samples: mapa nota MIDI -> sample (sha) para los instrumentos integrados.

Un kit es un asset JSON (tipo ``preset``) referenciado por ``PluginSlot.preset`` del instrumento
``drums`` (las notas mapeadas suenan con el sample; el resto con la batería sintética) o
``sampler`` (solo suenan las notas mapeadas). Al ser un asset direccionado por contenido, cambiar
el kit es registrar uno nuevo y reemplazar el plugin con una operación validada.

Reproducción: el sample entra en la muestra exacta de la nota (``tick_to_sample``), entero,
con ganancia ``velocidad/127 · 10^(gain_db/20)``; la duración de la nota no lo corta (one-shot).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

KIT_FORMAT = "osky-kit"
KIT_VERSION = 1
KIT_INSTRUMENTS = {"drums", "sampler"}
DEFAULT_PITCH = {"kick": 36, "snare": 38, "clap": 39, "hat": 42, "perc": 37}


def kit_bytes(voices: dict[int, dict]) -> bytes:
    """JSON canónico (mismo kit -> mismo sha)."""
    data = {"format": KIT_FORMAT, "version": KIT_VERSION,
            "voices": {str(int(p)): {"sample": v["sample"], "gain_db": float(v.get("gain_db", 0.0)),
                                     **({"name": v["name"]} if v.get("name") else {})}
                       for p, v in sorted(voices.items(), key=lambda kv: int(kv[0]))}}
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def parse_kit(raw: bytes | str) -> dict[int, dict]:
    data = json.loads(raw)
    if data.get("format") != KIT_FORMAT:
        raise ValueError("el preset no es un kit de samples (format != osky-kit)")
    if data.get("version") != KIT_VERSION:
        raise ValueError(f"versión de kit no soportada: {data.get('version')}")
    out = {}
    for p, v in data.get("voices", {}).items():
        pitch = int(p)
        if not 0 <= pitch <= 127:
            raise ValueError(f"nota fuera de rango en el kit: {pitch}")
        sha = v.get("sample", "")
        if not (isinstance(sha, str) and len(sha) == 64):
            raise ValueError(f"sample inválido para la nota {pitch}")
        out[pitch] = {"sample": sha, "gain_db": float(v.get("gain_db", 0.0)), "name": v.get("name")}
    return out


def load_kit(path: Path) -> dict[int, dict]:
    return parse_kit(Path(path).read_bytes())


def render_kit(notes, kit: dict[int, dict], load_audio, n: int) -> np.ndarray:
    """Suma los samples del kit en un búfer estéreo de ``n`` muestras.

    ``notes``: (muestra_inicio, muestra_fin, nota, velocidad); ``load_audio(sha)`` devuelve
    float32 (muestras, 2) ya a la frecuencia del proyecto.
    """
    y = np.zeros((n, 2), np.float32)
    audio: dict[str, np.ndarray] = {}
    for a, _b, p, v in notes:
        voice = kit.get(p)
        if voice is None or a >= n:
            continue
        sha = voice["sample"]
        if sha not in audio:
            audio[sha] = load_audio(sha)
        x = audio[sha]
        g = np.float32(v / 127 * 10 ** (voice["gain_db"] / 20))
        k = min(len(x), n - a)
        y[a:a + k] += x[:k] * g
    return y
