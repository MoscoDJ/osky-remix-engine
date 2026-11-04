"""Hallazgos del "doctor de mezcla" para el evaluador: enmascaramiento y referencia comercial.

- ``masking``: un enmascarador tapa a la voz (pad, lead, arp, chops) o el bajo al bombo en una
  banda de octava y sección (``analysis/masking.py``). Severidad 0.1 por dB sobre el umbral.
- ``reference_*``: diferencias con la referencia de ``meta.reference_track`` por encima de la
  tolerancia de la política (``analysis/reference_match.py``). Solo si hay referencia.
"""
from __future__ import annotations

import numpy as np

from ..analysis import masking as mk
from ..analysis import reference_match as rm
from ..core.cache import cache_key
from ..core.models import Project, TrackKind
from .evaluator import Finding
from .policies import Policy

# solo estos sentidos tienen acción (EQ en el enmascarador o ducking del bajo)
def _actionable(target: str, masker: str) -> bool:
    return target.startswith("vox") or (target == "kick" and masker == "bass")


def masking_findings(p: Project, post: dict, secs: list[dict], sr: int, policy: Policy) -> tuple[list[Finding], dict]:
    ids = [t.id for t in p.tracks if t.id in post and t.kind in (TrackKind.audio, TrackKind.instrument)]
    hits = mk.detect({t: post[t] for t in ids}, sr, secs, mk.masking_pairs(ids))
    groups = mk.summarize(hits, policy.masking_threshold_db)
    out = []
    for g in groups:
        if not _actionable(g["target"], g["masker"]):
            continue
        out.append(Finding("masking", round(0.1 * (g["masking_db"] - policy.masking_threshold_db), 3), g["detail"], g))
    return out, {"masking": [{k: g[k] for k in ("target", "masker", "band", "masking_db", "sections")} for g in groups]}


def reference_of(p: Project) -> str | None:
    ref = p.meta.get("reference_track")
    if isinstance(ref, dict):
        return ref.get("sha256") or ref.get("sha")
    return ref if isinstance(ref, str) else None


def tolerances(policy: Policy) -> dict:
    return {"band_db": policy.reference_band_tol_db, "width_db": policy.reference_width_tol_db,
            "lra_lu": policy.reference_lra_tol_lu, "plr_db": policy.reference_plr_tol_db,
            "punch_db": policy.reference_punch_tol_db}


def reference_profile(sha: str, assets, cache, sr: int) -> dict:
    """Perfil de la referencia (remuestreada a ``sr``), en caché por contenido."""
    key = cache_key("reference_profile", rm.VERSION, [sha], {"sr": sr})
    hit = cache.get_json(key) if cache is not None else None
    if hit is not None:
        return hit
    path = assets.find(sha)
    if path is None:
        raise FileNotFoundError(f"la referencia {sha[:12]} no está en el almacén")
    prof = rm.profile(rm.load_audio(path, sr), sr)
    if cache is not None:
        cache.put_json(key, prof)
    return prof


def reference_findings(p: Project, renderer, master_audio: np.ndarray, policy: Policy) -> tuple[list[Finding], dict]:
    sha = reference_of(p)
    if not sha:
        return [], {}
    try:
        pr = reference_profile(sha, renderer.assets, getattr(renderer, "cache", None), p.sample_rate)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        return [], {"reference_error": str(e)}
    rep = rm.match(master_audio, None, p.sample_rate, tol=tolerances(policy), ref_profile=pr)
    out = [Finding(f["code"], f["severity"], f["detail"], f) for f in rep["findings"]]
    return out, {"reference": {"sha256": sha, "tonal_delta_db": rep["tonal_delta_db"], **rep["dynamics"]}}
