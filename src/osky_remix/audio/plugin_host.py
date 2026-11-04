"""Host de plugins basado en DawDreamer.

Notas de compatibilidad verificadas (ver docs/COMPATIBILITY.md):
- Surge XT VST3 no acepta presets .fxp con ``load_preset`` (formato VST2). El estado
  VST3 de Surge es el mismo bloque ``sub3`` que va dentro del .fxp, así que se convierte
  el .fxp a un archivo de estado de DawDreamer y se carga con ``load_state``.
- Tras ``load_state`` Surge carga el patch de forma asíncrona: las notas que llegan antes
  de ~0.5 s se pierden. Todo render con Surge usa ``PREROLL_S`` y recorta después.
- ``load_state`` devuelve None (no indica éxito) y ``get_parameter`` devuelve la caché del
  host, que no se actualiza tras ``load_state``. Verificar estados siempre por audio.
- Un ``set_parameter`` hecho tras ``load_state`` lo pisa la carga asíncrona del patch.
  Política: los parámetros estáticos se aplican como automatización constante
  (``set_static_params``), que se reafirma en cada bloque.
"""
from __future__ import annotations

import struct
from pathlib import Path

SURGE_XT_VST3 = Path("/usr/lib/vst3/Surge XT.vst3")
SURGE_XT_FX_VST3 = Path("/usr/lib/vst3/Surge XT Effects.vst3")
SURGE_FACTORY = Path("/usr/share/surge-xt/patches_factory")
PREROLL_S = 0.5
# Determinismo (verificado por el doctor, procesos nuevos): algunos patches de Surge tienen
# aleatoriedad interna (p. ej. el patch por defecto, Keys/EP 1) y difieren en cada render;
# otros (Leads/Acidofil, Basses/Attacked) son bit-exactos. Las pistas con patches
# estocásticos se congelan como asset (SHA-256) para que re-renderizar el proyecto sea exacto.
DETERMINISTIC_PRESET = SURGE_FACTORY / "Leads" / "Acidofil.fxp"
STOCHASTIC_PRESET = SURGE_FACTORY / "Keys" / "EP 1.fxp"

# Alfabeto de MemoryBlock::toBase64Encoding de JUCE (no es base64 estándar).
_JUCE_B64 = ".ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+"


def juce_b64_encode(data: bytes) -> str:
    value = int.from_bytes(data, "little")
    n_chars = (len(data) * 8 + 5) // 6
    return f"{len(data)}." + "".join(_JUCE_B64[(value >> (6 * i)) & 63] for i in range(n_chars))


def juce_b64_decode(text: str) -> bytes:
    size_str, body = text.split(".", 1)
    size = int(size_str)
    value = 0
    for i, ch in enumerate(body):
        value |= _JUCE_B64.index(ch) << (6 * i)
    return value.to_bytes(size + 8, "little")[:size]


def fxp_chunk(fxp_path: Path) -> bytes:
    data = Path(fxp_path).read_bytes()
    if data[:4] != b"CcnK" or data[8:12] != b"FPCh":
        raise ValueError(f"{fxp_path} no es un .fxp de tipo chunk (FPCh)")
    size = struct.unpack(">I", data[56:60])[0]
    chunk = data[60:60 + size]
    if len(chunk) != size:
        raise ValueError(f"{fxp_path}: chunk truncado ({len(chunk)} de {size} bytes)")
    return chunk


def vst3_state_from_chunk(chunk: bytes) -> bytes:
    """Construye el archivo de estado que DawDreamer/JUCE espera para un VST3."""
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n\n<VST3PluginState><IComponent>'
           + juce_b64_encode(chunk) + "</IComponent></VST3PluginState>\n").encode() + b"\x00"
    return b"VC2!" + struct.pack("<I", len(xml) - 1) + xml


def fxp_to_state_file(fxp_path: Path, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.write_bytes(vst3_state_from_chunk(fxp_chunk(fxp_path)))
    return out_path


def set_static_params(processor, values: dict[str, float], total_seconds: float, sr: int) -> None:
    """Fija parámetros por nombre como automatización constante durante todo el render."""
    import numpy as np
    n = int(np.ceil(total_seconds * sr))
    for name, v in values.items():
        processor.set_automation(find_param(processor, name), np.full(n, float(v), np.float32))


def find_param(processor, name: str) -> int:
    for p in processor.get_parameters_description():
        if p["name"] == name:
            return int(p["index"])
    raise KeyError(f"parámetro no encontrado: {name}")
