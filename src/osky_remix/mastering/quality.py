"""Reporte de calidad del master: mediciones y verificaciones con estado PASS/WARN/FAIL."""
from __future__ import annotations

import numpy as np

from .loudness import _as_frames, integrated_lufs
from .true_peak import sample_peak_dbfs, true_peak_dbtp


def quality_report(master: np.ndarray, sr: int, ceiling_dbtp: float, target_lufs: float,
                   premaster: np.ndarray | None = None) -> dict:
    from ..analysis.features import loudness_range
    x = _as_frames(master).astype(np.float64)
    lufs = integrated_lufs(x, sr)
    tp = true_peak_dbtp(x, sr)
    checks = []

    def chk(name, ok, detail, warn_only=False):
        checks.append({"check": name, "status": "PASS" if ok else ("WARN" if warn_only else "FAIL"), "detail": detail})

    chk("true_peak", tp <= ceiling_dbtp + 1e-3, f"{tp:.2f} dBTP (techo {ceiling_dbtp})")
    chk("loudness", abs(lufs - target_lufs) <= 0.5, f"{lufs:.2f} LUFS (objetivo {target_lufs})")
    clipped = int(np.sum(np.abs(x) >= 0.999))
    chk("clipping", clipped == 0, f"{clipped} muestras ≥ 0.999")
    chk("finite", bool(np.isfinite(x).all()), "sin NaN/Inf")
    dc = float(np.abs(x.mean(0)).max())
    chk("dc_offset", dc < 1e-3, f"{dc:.5f}", warn_only=True)
    corr = mono_drop = None
    if x.shape[1] == 2:
        l, r = x[:, 0], x[:, 1]
        corr = float(np.corrcoef(l, r)[0, 1]) if np.std(l) > 0 and np.std(r) > 0 else 1.0
        m = (l + r) / 2
        mono_drop = lufs - integrated_lufs(np.stack([m, m], 1), sr)   # mono en ambos canales
        chk("mono_compat", mono_drop < 3.0, f"correlación {corr:.2f}, pérdida en mono {mono_drop:.2f} LU", warn_only=True)
    head = int(np.argmax(np.abs(x).max(1) > 10 ** (-60 / 20))) if np.any(np.abs(x) > 1e-3) else len(x)
    chk("leading_silence", head < 0.5 * sr, f"{head / sr:.2f} s", warn_only=True)
    lra = loudness_range(x, sr)
    rep = {"lufs": round(lufs, 2), "true_peak_dbtp": round(tp, 2), "sample_peak_dbfs": round(sample_peak_dbfs(x), 2),
           "loudness_range_lu": round(lra, 2), "stereo_correlation": None if corr is None else round(corr, 3),
           "mono_loss_lu": None if mono_drop is None else round(mono_drop, 2), "clipped_samples": clipped,
           "duration_s": round(len(x) / sr, 3), "checks": checks,
           "status": "FAIL" if any(c["status"] == "FAIL" for c in checks) else
                     ("WARN" if any(c["status"] == "WARN" for c in checks) else "PASS")}
    if premaster is not None:
        pm = _as_frames(premaster)
        rep["premaster"] = {"lufs": round(integrated_lufs(pm, sr), 2), "peak_dbfs": round(sample_peak_dbfs(pm), 2)}
        rep["limiting_gain_reduction_db_est"] = round((integrated_lufs(pm, sr) - lufs), 2)
    def finite(value):
        if isinstance(value, dict):
            return {k: finite(v) for k, v in value.items()}
        if isinstance(value, list):
            return [finite(v) for v in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value
    return finite(rep)
