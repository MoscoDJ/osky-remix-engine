"""Tipos comunes: toda estimación lleva valor, confianza (0-1), alternativas y método."""
from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

AN_SR = 22050          # frecuencia de trabajo interna del análisis
HOP = 256


class Alternative(BaseModel):
    value: Any
    score: float


class Estimate(BaseModel):
    value: Any
    confidence: float = Field(ge=0.0, le=1.0)
    alternatives: list[Alternative] = Field(default_factory=list)
    method: str
    notes: str = ""


def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x
    return x.mean(axis=1) if x.shape[1] <= 8 else x.mean(axis=0)


def resample_mono(x: np.ndarray, sr: int, target: int = AN_SR) -> np.ndarray:
    import librosa
    m = to_mono(x)
    return m if sr == target else librosa.resample(m, orig_sr=sr, target_sr=target)
