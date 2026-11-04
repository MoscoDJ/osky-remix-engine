"""Comparación de la mezcla con una referencia comercial, igualada en sonoridad.

Métricas (ver docs/AGENT.md, "Comparar con una referencia"):
- **Balance tonal**: espectro promedio a largo plazo por banda de octava (``bands.band_spectrum``).
  La diferencia referencia − mezcla se alinea por su **mediana** entre bandas: así una diferencia
  de sonoridad o una sola banda distinta no desplaza a las demás (igualar por LUFS movería todas
  las bandas, porque la ponderación K pesa más los agudos). Positivo = a la mezcla le falta.
- **Estéreo por banda**: lado/centro (dB) y correlación L/R en cada banda.
- **Dinámica**: LRA (aprox. EBU, ``features.loudness_range``), factor de cresta (pico de muestra /
  RMS) y PLR (true peak − LUFS integrada).
- **Pegada en graves**: en 40–120 Hz, ``punch_db`` = percentil 99 de la envolvente RMS de 10 ms
  menos el RMS de la banda, y ``transients_per_s`` = ataques por segundo (subida ≥ 6 dB en ≤ 30 ms,
  dentro de 20 dB del máximo, 100 ms de rearme).

Todas son independientes del nivel absoluto; la referencia se lleva igualmente a la sonoridad de
la mezcla (``ref_gain_to_match_db``) y se remuestrea a la frecuencia del proyecto.
"""
from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt

from ..mastering.loudness import integrated_lufs
from ..mastering.true_peak import true_peak_dbtp
from .bands import BANDS, as_stereo, band_label, band_spectrum
from .features import loudness_range

VERSION = "1"
PUNCH_BAND = (40.0, 120.0)
# por debajo de esto (relativo a la banda más fuerte de la referencia) la banda no se juzga
BAND_RELEVANCE_DB = -45.0
WIDTH_CLAMP = (-40.0, 10.0)

TOLERANCES = {"band_db": 2.0, "width_db": 4.0, "lra_lu": 3.0, "plr_db": 2.5, "punch_db": 3.0}


def load_audio(path: Path, sr: int) -> np.ndarray:
    """Lee cualquier archivo de audio a estéreo float64 en ``sr``."""
    x, sr0 = sf.read(str(path), dtype="float64", always_2d=True)
    x = as_stereo(x)
    if sr0 != sr:
        g = gcd(int(sr0), int(sr))
        x = resample_poly(x, sr // g, sr0 // g, axis=0)
    return x


def _lufs(x: np.ndarray, sr: int) -> float | None:
    try:
        v = integrated_lufs(x, sr)
    except ValueError:
        return None
    return float(v) if np.isfinite(v) else None


def punch(x: np.ndarray, sr: int) -> dict:
    m = as_stereo(x).mean(1)
    y = sosfiltfilt(butter(4, PUNCH_BAND, "band", fs=sr, output="sos"), m)
    hop = max(1, int(0.010 * sr))
    n = len(y) // hop
    if n < 4:
        return {"punch_db": 0.0, "transients_per_s": 0.0}
    env = 10 * np.log10(np.mean(y[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-20)
    rms = 10 * np.log10(np.mean(y ** 2) + 1e-20)
    top = float(np.percentile(env, 99))
    count, last = 0, -10 ** 9
    for i in range(3, n):
        if env[i] >= top - 20 and env[i] - env[i - 3: i].min() >= 6 and i - last >= 10:
            count += 1
            last = i
    return {"punch_db": round(top - rms, 2), "transients_per_s": round(count / (len(y) / sr), 3)}


def profile(x: np.ndarray, sr: int) -> dict:
    """Mediciones de una señal (estéreo o mono) a ``sr``."""
    x = as_stereo(x)
    lufs = _lufs(x, sr)
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.abs(x).max())
    tp = float(true_peak_dbtp(x, sr)) if peak > 0 else None
    return {"version": VERSION, "sr": sr, "duration_s": round(len(x) / sr, 3), "lufs": lufs,
            "bands": band_spectrum(x, sr), "lra_lu": round(loudness_range(x, sr), 2),
            "crest_db": round(20 * np.log10(peak / rms), 2) if rms > 0 else 0.0,
            "plr_db": round(tp - lufs, 2) if tp is not None and lufs is not None else None,
            "true_peak_dbtp": None if tp is None else round(tp, 2), **punch(x, sr)}


def _clamp_w(v: float) -> float:
    return float(np.clip(v, *WIDTH_CLAMP))


def compare_profiles(mix: dict, ref: dict, tol: dict | None = None) -> dict:
    """Deltas (referencia − mezcla) y hallazgos en lenguaje llano."""
    tol = {**TOLERANCES, **(tol or {})}
    findings: list[dict] = []
    top = max(b["db"] for b in ref["bands"])
    raw = {}
    for bm, br, meta in zip(mix["bands"], ref["bands"], BANDS):
        if br["db"] >= top + BAND_RELEVANCE_DB or bm["db"] >= top + BAND_RELEVANCE_DB:
            raw[meta["id"]] = br["db"] - bm["db"]
    offset = float(np.median(list(raw.values()))) if raw else 0.0
    tonal = {k: round(v - offset, 2) for k, v in raw.items()}
    width = {}
    for bm, br, meta in zip(mix["bands"], ref["bands"], BANDS):
        if meta["id"] not in tonal:
            continue
        wm, wr = _clamp_w(bm["side_to_mid_db"]), _clamp_w(br["side_to_mid_db"])
        width[meta["id"]] = {"mix": round(wm, 2), "ref": round(wr, 2), "delta_db": round(wr - wm, 2),
                             "corr_mix": round(bm["corr"], 3), "corr_ref": round(br["corr"], 3)}
    for meta in BANDS:
        d = tonal.get(meta["id"])
        if d is not None and abs(d) > tol["band_db"]:
            lab = band_label(meta)
            findings.append({"code": "reference_tonal", "band": meta["id"], "fc": meta["fc"], "lo": meta["lo"],
                             "hi": meta["hi"], "delta_db": d, "severity": round(0.25 * (abs(d) - tol["band_db"]), 3),
                             "detail": (f"faltan ~{d:.1f} dB entre {lab} respecto a la referencia" if d > 0 else
                                        f"sobran ~{-d:.1f} dB entre {lab} respecto a la referencia")})
        w = width.get(meta["id"])
        if w and abs(w["delta_db"]) > tol["width_db"]:
            lab = band_label(meta)
            more = "más estrecha" if w["delta_db"] > 0 else "más ancha"
            findings.append({"code": "reference_width", "band": meta["id"], "delta_db": w["delta_db"],
                             "severity": round(0.1 * (abs(w["delta_db"]) - tol["width_db"]), 3),
                             "detail": f"la mezcla es {more} que la referencia en {lab} (lado/centro "
                                       f"{w['mix']:+.1f} vs {w['ref']:+.1f} dB)"})
    dyn = {"lra_delta_lu": round(ref["lra_lu"] - mix["lra_lu"], 2),
           "crest_delta_db": round(ref["crest_db"] - mix["crest_db"], 2),
           "plr_delta_db": (round(ref["plr_db"] - mix["plr_db"], 2)
                            if ref.get("plr_db") is not None and mix.get("plr_db") is not None else None),
           "punch_delta_db": round(ref["punch_db"] - mix["punch_db"], 2),
           "transients_delta_per_s": round(ref["transients_per_s"] - mix["transients_per_s"], 3)}
    d = dyn["lra_delta_lu"]
    if abs(d) > tol["lra_lu"]:
        findings.append({"code": "reference_dynamics", "metric": "lra", "delta": d,
                         "severity": round(0.2 * (abs(d) - tol["lra_lu"]), 3),
                         "detail": f"la referencia tiene {'más' if d > 0 else 'menos'} rango de sonoridad "
                                   f"(LRA {ref['lra_lu']:.1f} vs {mix['lra_lu']:.1f} LU)"})
    d = dyn["plr_delta_db"]
    if d is not None and abs(d) > tol["plr_db"]:
        findings.append({"code": "reference_dynamics", "metric": "plr", "delta": d,
                         "severity": round(0.2 * (abs(d) - tol["plr_db"]), 3),
                         "detail": f"la mezcla está {'más' if d > 0 else 'menos'} comprimida que la referencia "
                                   f"(PLR {mix['plr_db']:.1f} vs {ref['plr_db']:.1f} dB)"})
    d = dyn["punch_delta_db"]
    if abs(d) > tol["punch_db"]:
        findings.append({"code": "reference_punch", "metric": "punch", "delta": d,
                         "severity": round(0.1 * (abs(d) - tol["punch_db"]), 3),
                         "detail": f"los graves (40–120 Hz) tienen {'menos' if d > 0 else 'más'} pegada que la "
                                   f"referencia (pico/RMS {mix['punch_db']:.1f} vs {ref['punch_db']:.1f} dB)"})
    return {"tonal_delta_db": tonal, "tonal_offset_db": round(offset, 2), "width": width, "dynamics": dyn,
            "findings": findings}


def match(mix: np.ndarray, ref: np.ndarray, sr: int, sections: list[tuple[str, int, int]] | None = None,
          tol: dict | None = None, ref_profile: dict | None = None) -> dict:
    """Reporte completo mezcla vs referencia (ambas a ``sr``)."""
    mix = as_stereo(mix)
    pm = profile(mix, sr)
    pr = ref_profile or profile(ref, sr)
    rep = {"mix": pm, "reference": pr,
           "ref_gain_to_match_db": (round(pm["lufs"] - pr["lufs"], 2)
                                    if pm["lufs"] is not None and pr["lufs"] is not None else None),
           **compare_profiles(pm, pr, tol)}
    if sections:
        rep["sections"] = []
        for name, a, b in sections:
            seg = mix[a:b]
            if len(seg) < sr:
                continue
            ps = {"bands": band_spectrum(seg, sr)}
            top = max(x["db"] for x in pr["bands"])
            raw = {m["id"]: br["db"] - bs["db"] for bs, br, m in zip(ps["bands"], pr["bands"], BANDS)
                   if br["db"] >= top + BAND_RELEVANCE_DB}
            off = float(np.median(list(raw.values()))) if raw else 0.0
            rep["sections"].append({"section": name, "tonal_delta_db": {k: round(v - off, 2) for k, v in raw.items()}})
    return rep


def report_text(rep: dict) -> str:
    L = [f"Mezcla {rep['mix']['lufs']} LUFS · referencia {rep['reference']['lufs']} LUFS "
         f"(referencia igualada {rep['ref_gain_to_match_db']:+.2f} dB)" if rep["ref_gain_to_match_db"] is not None
         else "Sonoridad no medible", "", "Banda              Δ tonal   lado/centro mezcla→ref   corr mezcla→ref"]
    for m in BANDS:
        if m["id"] in rep["tonal_delta_db"]:
            w = rep["width"][m["id"]]
            L.append(f"  {band_label(m):16s} {rep['tonal_delta_db'][m['id']]:+6.1f} dB   {w['mix']:+6.1f} → {w['ref']:+6.1f} dB"
                     f"         {w['corr_mix']:+.2f} → {w['corr_ref']:+.2f}")
    mx, rf = rep["mix"], rep["reference"]
    L += ["", f"LRA {mx['lra_lu']} → {rf['lra_lu']} LU · cresta {mx['crest_db']} → {rf['crest_db']} dB · "
              f"PLR {mx['plr_db']} → {rf['plr_db']} dB",
          f"Pegada 40–120 Hz {mx['punch_db']} → {rf['punch_db']} dB · ataques {mx['transients_per_s']} → "
          f"{rf['transients_per_s']} /s", ""]
    L += [f"• {f['detail']}" for f in rep["findings"]] or ["Sin diferencias por encima de la tolerancia."]
    return "\n".join(L)
