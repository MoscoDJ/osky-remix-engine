"""Comparación A/B igualada en sonoridad.

Juzgar dos mezclas a distinto volumen engaña (la más fuerte "suena mejor"): se iguala B a la
sonoridad integrada de A y se comparan bandas, secciones, estéreo, dinámica y un "null test".
"""
from __future__ import annotations

import numpy as np

from ..analysis.features import BANDS
from .loudness import _as_frames, integrated_lufs


def _bands_db(x: np.ndarray, sr: int) -> dict[str, float]:
    m = x.mean(1)
    n = 1 << int(np.floor(np.log2(max(2, min(len(m), sr * 30)))))   # potencia de 2 que cabe
    hop = n // 2
    acc = None
    for a in range(0, max(1, len(m) - n + 1), hop):
        s = np.abs(np.fft.rfft(m[a:a + n] * np.hanning(n))) ** 2
        acc = s if acc is None else acc + s
    f = np.fft.rfftfreq(n, 1 / sr)
    return {k: float(10 * np.log10(acc[(f >= lo) & (f < hi)].sum() + 1e-20)) for k, (lo, hi) in BANDS.items()}


def _lufs_safe(x: np.ndarray, sr: int) -> float:
    try:
        v = integrated_lufs(x, sr)
        return v if np.isfinite(v) else -70.0
    except ValueError:
        return -70.0


def compare(a: np.ndarray, b: np.ndarray, sr: int, sections: list[tuple[str, int, int]] | None = None) -> dict:
    a, b = _as_frames(a).astype(np.float64), _as_frames(b).astype(np.float64)
    la, lb = integrated_lufs(a, sr), integrated_lufs(b, sr)
    match_db = la - lb
    bm = b * 10 ** (match_db / 20)
    ba, bb = _bands_db(a, sr), _bands_db(bm, sr)
    rep = {"lufs_a": round(la, 2), "lufs_b": round(lb, 2), "b_gain_to_match_db": round(match_db, 2),
           "bands_db_b_minus_a": {k: round(bb[k] - ba[k], 2) for k in ba},
           "crest_db": {"a": round(_crest(a), 2), "b": round(_crest(bm), 2)}}
    if a.shape[1] == 2 and b.shape[1] == 2:
        rep["side_to_mid_db"] = {"a": round(_s2m(a), 2), "b": round(_s2m(bm), 2)}
    if len(a) == len(b):
        res = a - bm
        rep["null_residual_db"] = round(10 * np.log10(np.sum(res ** 2) / max(np.sum(a ** 2), 1e-20) + 1e-20), 2)
    if sections:
        rep["sections"] = []
        for name, s, e in sections:
            rep["sections"].append({"section": name, "lufs_a": round(_lufs_safe(a[s:e], sr), 2),
                                    "lufs_b_matched": round(_lufs_safe(bm[s:e], sr), 2)})
    return rep, bm.astype(np.float32)


def _crest(x: np.ndarray) -> float:
    rms = np.sqrt(np.mean(x ** 2))
    return float(20 * np.log10(np.abs(x).max() / rms)) if rms > 0 else 0.0


def _s2m(x: np.ndarray) -> float:
    mid, side = (x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2
    return float(10 * np.log10((np.mean(side ** 2) + 1e-20) / (np.mean(mid ** 2) + 1e-20)))


def report_markdown(rep: dict, diff: list[str], label_a: str, label_b: str) -> str:
    L = [f"# A/B: {label_a} vs {label_b}", "",
         f"Sonoridad: A {rep['lufs_a']} LUFS, B {rep['lufs_b']} LUFS → B ajustado {rep['b_gain_to_match_db']:+.2f} dB "
         "para comparar a igual volumen.", "", "## Balance por banda (B − A, igualado)", "",
         "| Banda | Δ dB |", "|---|---|"] + [f"| {k} | {v:+.2f} |" for k, v in rep["bands_db_b_minus_a"].items()]
    L += ["", f"Dinámica (cresta): A {rep['crest_db']['a']} dB, B {rep['crest_db']['b']} dB"]
    if "side_to_mid_db" in rep:
        L.append(f"Estéreo (lado/centro): A {rep['side_to_mid_db']['a']} dB, B {rep['side_to_mid_db']['b']} dB")
    if "null_residual_db" in rep:
        L.append(f"Null test (A − B igualado): {rep['null_residual_db']} dB respecto de A")
    if rep.get("sections"):
        L += ["", "## Por sección (igualado)", "", "| Sección | A LUFS | B LUFS |", "|---|---|---|"]
        L += [f"| {s['section']} | {s['lufs_a']} | {s['lufs_b_matched']} |" for s in rep["sections"]]
    L += ["", "## Cambios en el proyecto", ""] + ([f"- {d}" for d in diff] or ["- (ninguno)"])
    return "\n".join(L) + "\n"
