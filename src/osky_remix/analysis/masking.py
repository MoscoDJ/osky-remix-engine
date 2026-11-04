"""Detector de enmascaramiento en frecuencia entre salidas post-fader, por sección y banda.

Para un par (objetivo T, enmascarador M), en cada sección y banda de octava (``bands.BANDS``):

1. Energía por cuadro (STFT Hann de 4096, salto 2048) de T y de M en la banda.
2. La banda cuenta para T si su nivel (percentil 95 de los cuadros) está a ≤ 20 dB de la banda más
   fuerte de T en la sección: no se juzga a la voz donde no tiene contenido.
3. Cuadros activos de T: a ≤ 15 dB de ese percentil 95 (en el bombo, los golpes; en la voz, las
   frases). Se exigen al menos 4.
4. ``masking_db`` = mediana, sobre los cuadros activos, de (energía de M − energía de T).
   0 dB = M tan fuerte como T en esa banda mientras T suena; se marca si supera el umbral
   (``Policy.masking_threshold_db``, −3 dB por defecto).

Pares: la voz (pistas ``vox*``) frente a pad, lead, arp y chops; y bajo↔bombo en ambos sentidos
(el reporte muestra los dos; solo bajo→bombo genera hallazgo accionable).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bands import BANDS, band_frames, band_label, frame_range

VOCAL_MASKERS = ("pad", "lead", "arp", "chops")
RELEVANCE_DB = 20.0
ACTIVE_DB = 15.0
MIN_FRAMES = 4
ABS_FLOOR_DB = -90.0


@dataclass
class MaskingHit:
    target: str
    masker: str
    section: str
    band: str
    fc: float
    masking_db: float
    active_frames: int

    def label(self) -> str:
        b = next(x for x in BANDS if x["id"] == self.band)
        return band_label(b)


def masking_pairs(track_ids: list[str]) -> list[tuple[str, str]]:
    """(objetivo, enmascarador) que interesa vigilar."""
    vox = [t for t in track_ids if t.startswith("vox")]
    out = [(v, m) for v in vox for m in track_ids if m.startswith(VOCAL_MASKERS) and m != v]
    if "kick" in track_ids and "bass" in track_ids:
        out += [("kick", "bass"), ("bass", "kick")]
    return out


def measure_pair(ft: np.ndarray, fm: np.ndarray) -> list[tuple[int, float, int]]:
    """(banda, masking_db, cuadros activos) para las bandas relevantes de T. ``ft``/``fm``: (cuadros, bandas)."""
    if len(ft) == 0:
        return []
    p95 = np.percentile(ft, 95, axis=0)
    top = p95.max()
    out = []
    for b in range(ft.shape[1]):
        if p95[b] < top - RELEVANCE_DB or p95[b] < ABS_FLOOR_DB:
            continue
        act = ft[:, b] >= p95[b] - ACTIVE_DB
        if act.sum() < MIN_FRAMES:
            continue
        out.append((b, float(np.median(fm[act, b] - ft[act, b])), int(act.sum())))
    return out


def detect(post: dict[str, np.ndarray], sr: int, sections: list[dict], pairs: list[tuple[str, str]] | None = None,
           n_fft: int = 4096, hop: int = 2048) -> list[MaskingHit]:
    """Todas las mediciones (marcadas o no) por par, sección y banda relevante.

    ``sections``: dicts con ``name``, ``start``, ``end`` (muestras) y opcionalmente ``vocal``; los
    pares de voz solo se miden en secciones con voz (si la sección no lo declara, se mide).
    """
    pairs = pairs if pairs is not None else masking_pairs(list(post))
    need = {t for pr in pairs for t in pr if t in post}
    frames = {t: band_frames(post[t], sr, n_fft, hop) for t in need}
    hits = []
    for tgt, msk in pairs:
        if tgt not in frames or msk not in frames:
            continue
        nf = min(len(frames[tgt]), len(frames[msk]))
        for s in sections:
            if tgt.startswith("vox") and not s.get("vocal", True):
                continue
            sl = frame_range(s["start"], s["end"], nf, n_fft, hop)
            for b, m, n in measure_pair(frames[tgt][sl], frames[msk][sl]):
                hits.append(MaskingHit(tgt, msk, s["name"], BANDS[b]["id"], BANDS[b]["fc"], round(m, 2), n))
    return hits


def summarize(hits: list[MaskingHit], threshold_db: float) -> list[dict]:
    """Agrupa por (objetivo, enmascarador, banda): peor sección y lista de secciones afectadas."""
    groups: dict[tuple[str, str, str], list[MaskingHit]] = {}
    for h in hits:
        if h.masking_db > threshold_db:
            groups.setdefault((h.target, h.masker, h.band), []).append(h)
    out = []
    for (tgt, msk, band), hs in groups.items():
        worst = max(hs, key=lambda h: h.masking_db)
        out.append({"target": tgt, "masker": msk, "band": band, "fc": worst.fc, "masking_db": worst.masking_db,
                    "section": worst.section, "sections": [h.section for h in hs],
                    "detail": f"'{msk}' enmascara a '{tgt}' en {worst.label()} "
                              f"({worst.masking_db:+.1f} dB respecto de '{tgt}' en '{worst.section}'"
                              + (f" y {len(hs) - 1} sección(es) más" if len(hs) > 1 else "") + ")"})
    return sorted(out, key=lambda d: -d["masking_db"])
