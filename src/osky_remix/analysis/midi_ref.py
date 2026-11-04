"""MIDI de referencia (p. ej. karaoke): lectura, alineación con la grabación y extracción de
armonía, contramelodías y letra con tiempos.

Un MIDI de terceros no coincide con la grabación (otra introducción, tempo ligeramente distinto),
así que se alinea con DTW sobre croma armónico (MIDI sin batería vs. stems bajo+otros del audio).
La alineación se verifica con la batería del MIDI contra la rejilla de beats del audio, y con la
letra contra la actividad vocal.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mido
import numpy as np
from pydantic import BaseModel

from .common import AN_SR, clip01
from .harmony import NOTE, _templates

HOP = 4096                        # ≈ 0.186 s a 22.05 kHz


@dataclass
class RawNote:
    start: float                  # s (tiempo del MIDI)
    end: float
    pitch: int
    velocity: int
    channel: int


@dataclass
class RawMidi:
    notes: list[RawNote]
    programs: dict[int, int]
    lyrics: list[tuple[float, str]] = field(default_factory=list)
    tempos: list[tuple[float, float]] = field(default_factory=list)
    meters: list[tuple[float, str]] = field(default_factory=list)
    beat_times: list[float] = field(default_factory=list)      # cada negra del MIDI, en segundos


def read_midi(path) -> RawMidi:
    mf = mido.MidiFile(str(path))
    events = []
    for tr in mf.tracks:                                  # tipo 0 o 1: se fusionan por tick absoluto
        tick = 0
        for msg in tr:
            tick += msg.time
            events.append((tick, msg))
    events.sort(key=lambda e: e[0])
    tempo, t, last_tick = 500000, 0.0, 0
    on: dict[tuple[int, int], tuple[float, int]] = {}
    out = RawMidi(notes=[], programs={})
    ppq = mf.ticks_per_beat
    next_beat = 0
    for tick, msg in events:
        while next_beat <= tick:                          # negras hasta este evento, con el tempo vigente
            out.beat_times.append(t + mido.tick2second(next_beat - last_tick, ppq, tempo))
            next_beat += ppq
        t += mido.tick2second(tick - last_tick, ppq, tempo)
        last_tick = tick
        if msg.type == "set_tempo":
            tempo = msg.tempo
            out.tempos.append((t, round(mido.tempo2bpm(msg.tempo), 3)))
        elif msg.type == "time_signature":
            out.meters.append((t, f"{msg.numerator}/{msg.denominator}"))
        elif msg.type == "program_change":
            out.programs[msg.channel] = msg.program
        elif msg.type in ("lyrics", "text"):
            out.lyrics.append((t, msg.text))
        elif msg.type == "note_on" and msg.velocity > 0:
            on[(msg.channel, msg.note)] = (t, msg.velocity)
        elif msg.type in ("note_off", "note_on") and (msg.channel, msg.note) in on:
            s, v = on.pop((msg.channel, msg.note))
            out.notes.append(RawNote(s, t, msg.note, v, msg.channel))
    return out


ROLE_BY_PROGRAM = [(range(0, 8), "piano"), (range(16, 24), "organ"), (range(24, 32), "guitar"),
                   (range(32, 40), "bass"), (range(40, 52), "strings"), (range(52, 56), "voice"),
                   (range(56, 64), "brass"), (range(64, 80), "winds"), (range(80, 104), "synth")]


def channel_roles(raw: RawMidi) -> dict[int, str]:
    roles = {}
    for ch in sorted({n.channel for n in raw.notes}):
        if ch == 9:
            roles[ch] = "drums"
            continue
        prog = raw.programs.get(ch, 0)
        roles[ch] = next((r for rng, r in ROLE_BY_PROGRAM if prog in rng), "other")
    return roles


def lyric_lines(raw: RawMidi) -> list[tuple[float, float, str]]:
    """Líneas (inicio, fin, texto). '/' = nueva línea, '\\' = párrafo (convención .kar)."""
    lines, cur, t0 = [], "", None
    for t, txt in raw.lyrics:
        if txt.startswith(("/", "\\")) and cur.strip():
            lines.append([t0, t, cur.strip()])
            cur, t0 = "", None
        piece = txt.lstrip("/\\")
        if piece and t0 is None:
            t0 = t
        cur += piece
    if cur.strip():
        lines.append([t0, raw.lyrics[-1][0] + 1.0, cur.strip()])
    out = []
    for i, (a, b, s) in enumerate(lines):
        if any(c in s for c in "=*") or a is None:
            continue                                     # títulos del karaoke
        end = min(b, lines[i + 1][0]) if i + 1 < len(lines) else b
        out.append((float(a), float(end), s))
    return out


def midi_chroma(raw: RawMidi, roles: dict[int, str], n_frames: int, frame_s: float) -> np.ndarray:
    C = np.zeros((12, n_frames))
    for n in raw.notes:
        if roles.get(n.channel) in ("drums", "voice"):
            continue
        a, b = int(n.start / frame_s), int(np.ceil(n.end / frame_s))
        w = n.velocity / 127 * (1.5 if roles.get(n.channel) == "bass" else 1.0)
        C[n.pitch % 12, max(0, a):min(n_frames, max(a + 1, b))] += w
    return C


class RefNote(BaseModel):
    start: float                 # s, tiempo de la GRABACIÓN
    end: float
    pitch: int
    velocity: int


class RefChannel(BaseModel):
    channel: int
    program: int
    role: str
    notes: list[RefNote]


class RefLyric(BaseModel):
    start: float                 # s, tiempo de la grabación
    end: float
    text: str


class Reference(BaseModel):
    version: str = "1"
    alignment_similarity: float  # coseno medio sobre el camino de DTW
    drum_error_ms: float | None  # mediana |golpe MIDI alineado − ataque más cercano del stem de batería|
    lyric_vocal_overlap: float | None
    chord_agreement: float | None
    midi_tempo: list[tuple[float, float]]
    midi_meter: list[tuple[float, str]]
    channels: list[RefChannel]
    lyrics: list[RefLyric]
    chords: list[tuple[int, int, str]]   # (muestra ini, muestra fin, acorde) en la grabación
    beat_anchors: int = 0                # negras del MIDI ancladas a beats del audio
    vocal_fit: dict[int, float] = {}     # juez: voz original en la armonía del MIDI, por desfase en tiempos
    confidence: float


def _activity(x: np.ndarray, k: int = 5) -> np.ndarray:
    x = np.convolve(x, np.ones(k) / k, mode="same")
    return x / (np.percentile(x, 95) + 1e-9)


def align(raw: RawMidi, harm_audio: np.ndarray, sr: int, drums_audio: np.ndarray | None = None,
          drum_weight: float = 0.3) -> tuple[np.ndarray, np.ndarray, float]:
    """Devuelve (t_midi, t_audio) monótonos para interpolar, y la similitud media del camino.

    Costo = croma armónico; si hay stem de batería, se suma la actividad de batería (golpes del MIDI
    vs. ataques del stem): marca entradas de sección que la armonía sola no distingue."""
    import librosa
    from .common import resample_mono
    y = resample_mono(harm_audio, sr)
    A = librosa.feature.chroma_cqt(y=y, sr=AN_SR, hop_length=HOP)
    frame_s = HOP / AN_SR
    roles = channel_roles(raw)
    midi_len = max(n.end for n in raw.notes)
    M = midi_chroma(raw, roles, int(midi_len / frame_s) + 1, frame_s)
    norm = lambda X: X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-9)
    An, Mn = norm(A), norm(M)
    cost = 1 - Mn.T @ An                                   # (midi, audio)
    if drums_audio is not None and any(n.channel == 9 for n in raw.notes):
        yd = resample_mono(drums_audio, sr)
        da = librosa.onset.onset_strength(y=yd, sr=AN_SR, hop_length=HOP)[: A.shape[1]]
        da = np.pad(da, (0, A.shape[1] - len(da)))
        dm = np.zeros(M.shape[1])
        for n in raw.notes:
            if n.channel == 9:
                dm[min(len(dm) - 1, int(n.start / frame_s))] += n.velocity / 127
        da, dm = np.clip(_activity(da), 0, 1.5), np.clip(_activity(dm), 0, 1.5)
        cost = (1 - drum_weight) * cost + drum_weight * np.abs(dm[:, None] - da[None, :]) / 1.5
    D, wp = librosa.sequence.dtw(C=cost, step_sizes_sigma=np.array([[1, 1], [1, 0], [0, 1], [1, 2], [2, 1]]),
                                 weights_add=np.array([0, 0.15, 0.15, 0, 0]), subseq=False)
    wp = wp[::-1]
    sim = float(np.mean([1 - cost[i, j] for i, j in wp]))
    tm = np.array([i for i, _ in wp]) * frame_s
    ta = np.array([j for _, j in wp]) * frame_s
    # un tiempo de MIDI -> mediana de los tiempos de audio emparejados; monótono y suavizado
    um = np.unique(tm)
    ua = np.array([np.median(ta[tm == u]) for u in um])
    ua = np.maximum.accumulate(ua)
    k = 9
    if len(ua) > k:
        pad = np.pad(ua, k // 2, mode="edge")
        ua = np.convolve(pad, np.ones(k) / k, mode="valid")
        ua = np.maximum.accumulate(ua)
    return um, ua, sim


def chords_from_notes(chans: list[RefChannel], beats: list[int], sr: int) -> list[tuple[int, int, str]]:
    """Acorde por beat de la grabación a partir de las notas alineadas (sin batería, voz ni metales)."""
    T, names = _templates()
    acc = [c for c in chans if c.role in ("bass", "guitar", "organ", "piano", "strings", "synth")]
    out: list[list] = []
    for i in range(len(beats) - 1):
        a, b = beats[i] / sr, beats[i + 1] / sr
        v = np.zeros(12)
        for c in acc:
            w = 1.5 if c.role == "bass" else 1.0
            for n in c.notes:
                ov = min(n.end, b) - max(n.start, a)
                if ov > 0:
                    v[n.pitch % 12] += w * ov * n.velocity / 127
        if v.sum() <= 0:
            lab = "N"
        else:
            lab = names[int(np.argmax(T @ (v / np.linalg.norm(v))))]
        if out and out[-1][2] == lab:
            out[-1][1] = beats[i + 1]
        else:
            out.append([beats[i], beats[i + 1], lab])
    return [tuple(x) for x in out]


def refine_with_beats(raw: RawMidi, tm: np.ndarray, ta: np.ndarray, beats_s: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Afina el mapa DTW (resolución ≈ 0.19 s) anclando cada negra del MIDI (donde hay batería) al
    beat del audio más cercano a donde la sitúa el DTW, si está a menos de 0.35 beat.

    Se probaron alternativas (voto por mayoría del desfase j−i; Viterbi sobre desfases con costo
    armónico) y las dos empeoraron la canción real según el juez independiente (voz original vs.
    armonía del MIDI, ver ``vocal_fit``): 68.4 % → 64 %. Se conserva la regla simple.
    """
    if len(beats_s) < 4 or not raw.beat_times:
        return tm, ta, 0
    period = float(np.median(np.diff(beats_s)))
    drum = [n.start for n in raw.notes if n.channel == 9]
    lo, hi = (min(drum), max(drum)) if drum else (0.0, float("inf"))
    pairs, used = [], set()
    region = [i for i, b in enumerate(raw.beat_times) if lo - 0.5 <= b <= hi + 0.5]
    for i in region:
        guess = float(np.interp(raw.beat_times[i], tm, ta))
        j = int(np.argmin(np.abs(beats_s - guess)))
        if abs(beats_s[j] - guess) <= 0.35 * period and j not in used:
            pairs.append((i, j))
            used.add(j)
    pairs.sort()
    mono = []
    for i, j in pairs:
        if not mono or (i > mono[-1][0] and j > mono[-1][1]):
            mono.append((i, j))
    # completar sin tocar anclas: 1 negra = 1 beat hacia atrás desde la primera ancla (hasta donde
    # empieza la batería del MIDI) y en huecos internos que cuadran exactamente. El DTW se adelanta
    # tras pasajes armónicamente estáticos y dejaba sin ancla justo la entrada de la batería.
    if mono:
        i0, j0 = mono[0]
        pre = [(i, j0 - (i0 - i)) for i in region if i < i0 and j0 - (i0 - i) >= 0]
        filled = pre + [mono[0]]
        for (i1, j1), (i2, j2) in zip(mono, mono[1:]):
            if i2 - i1 == j2 - j1:
                filled += [(i1 + d, j1 + d) for d in range(1, i2 - i1)]
            filled.append((i2, j2))
        mono = filled
    clean = []
    for i, j in mono:                                     # estrictamente creciente en ambos ejes
        m, a = raw.beat_times[i], float(beats_s[j])
        if not clean or (m > clean[-1][0] and a > clean[-1][1]):
            clean.append((m, a))
    if len(clean) < 8:
        return tm, ta, 0
    am, aa = np.array(clean).T
    keep = (tm < am[0]) | (tm > am[-1])                   # fuera de las anclas se conserva el DTW
    nm = np.concatenate([tm[keep], am])
    na = np.concatenate([ta[keep], aa])
    o = np.argsort(nm)
    return nm[o], np.maximum.accumulate(na[o]), len(clean)


def vocal_fit(channels: list["RefChannel"], vocal_notes: list[tuple[float, float, int]], period: float) -> dict[int, float]:
    """Juez independiente de la alineación: fracción de la voz original (melodía, en la grabación)
    que cae en notas de la armonía del MIDI (alineada), desplazando ±1 y ±2 tiempos. Bien alineado,
    el máximo está en 0 (o −1 por anticipación melódica)."""
    acc = [n for c in channels if c.role in ("bass", "guitar", "strings", "organ", "piano") for n in c.notes]
    if not acc or not vocal_notes:
        return {}
    st = np.array([n.start for n in acc]); en = np.array([n.end for n in acc]); pc = np.array([n.pitch % 12 for n in acc])
    out = {}
    for sh in (-2, -1, 0, 1, 2):
        inn = tot = 0.0
        for s0, e0, p in vocal_notes:
            t = (s0 + e0) / 2 + sh * period
            on = pc[(st <= t) & (en > t)]
            if len(on):
                d = e0 - s0
                tot += d
                inn += d * (p % 12 in set(on.tolist()))
        out[sh] = round(inn / tot, 4) if tot else 0.0
    return out


def build_reference(raw: RawMidi, harm_audio: np.ndarray, sr: int, beats: list[int],
                    audio_chords: list[tuple[int, int, str]] | None = None,
                    vocal_regions: list[tuple[int, int]] | None = None,
                    drums_audio: np.ndarray | None = None,
                    vocal_notes: list[tuple[float, float, int]] | None = None) -> Reference:
    tm, ta, sim = align(raw, harm_audio, sr, drums_audio)
    tm, ta, anchors = refine_with_beats(raw, tm, ta, np.array(beats) / sr)
    f = lambda t: float(np.interp(t, tm, ta))
    roles = channel_roles(raw)
    chans: dict[int, RefChannel] = {}
    for n in raw.notes:
        c = chans.setdefault(n.channel, RefChannel(channel=n.channel, program=raw.programs.get(n.channel, 0),
                                                  role=roles[n.channel], notes=[]))
        s, e = f(n.start), f(n.end)
        c.notes.append(RefNote(start=round(s, 4), end=round(max(e, s + 0.02), 4), pitch=n.pitch, velocity=n.velocity))
    # verificación 1 (independiente de las anclas, que usan la rejilla de beats): bombo y caja del
    # MIDI alineados contra los ATAQUES del stem de batería
    drum_err = None
    if drums_audio is not None:
        from .features import detect_transients
        ons = np.array([x.sample for x in detect_transients(drums_audio, sr) if x.strength > 0.15]) / sr
        hits = [n.start for n in chans.get(9, RefChannel(channel=9, program=0, role="drums", notes=[])).notes
                if n.pitch in (35, 36, 38, 40)]
        if len(ons) and hits:
            drum_err = float(np.median([np.min(np.abs(ons - h)) for h in hits]) * 1000)
    lyr = [RefLyric(start=round(f(a), 3), end=round(f(b), 3), text=s) for a, b, s in lyric_lines(raw)]
    # verificación 2: la letra cae donde hay voz
    overlap = None
    if vocal_regions and lyr:
        tot = sum(l.end - l.start for l in lyr)
        inside = 0.0
        for l in lyr:
            for a, b in vocal_regions:
                inside += max(0.0, min(l.end, b / sr) - max(l.start, a / sr))
        overlap = inside / tot if tot > 0 else None
    channels = sorted(chans.values(), key=lambda c: c.channel)
    chords = chords_from_notes(channels, beats, sr)
    agree = None
    if audio_chords:
        same = tot = 0
        for a, b, lab in chords:
            if lab == "N":
                continue
            mid = (a + b) // 2
            al = next((x for s0, e0, x in audio_chords if s0 <= mid < e0), "N")
            if al != "N":
                tot += 1
                same += al == lab
        agree = same / tot if tot else None
    fit = vocal_fit(channels, vocal_notes or [], float(np.median(np.diff(beats))) / sr) if vocal_notes else {}
    conf = clip01(0.4 * clip01((sim - 0.5) / 0.4) + 0.3 * (np.exp(-(drum_err or 200) / 60))
                  + 0.3 * (overlap if overlap is not None else 0.5))
    if fit and max(fit, key=fit.get) not in (0, -1):
        conf *= 0.5                                       # el juez independiente no respalda la alineación
    return Reference(alignment_similarity=round(sim, 3), drum_error_ms=None if drum_err is None else round(drum_err, 1),
                     lyric_vocal_overlap=None if overlap is None else round(overlap, 3),
                     chord_agreement=None if agree is None else round(agree, 3),
                     midi_tempo=raw.tempos, midi_meter=raw.meters, channels=channels, lyrics=lyr,
                     chords=chords, beat_anchors=anchors, vocal_fit=fit, confidence=round(conf, 3))


def find_lyric(ref: Reference, text: str, occurrence: int = 1) -> RefLyric | None:
    """Línea que contiene ``text`` (sin acentos ni mayúsculas); ``occurrence`` = n-ésima aparición."""
    import unicodedata
    norm = lambda s: "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")
    hits = [l for l in ref.lyrics if norm(text) in norm(l.text)]
    return hits[occurrence - 1] if len(hits) >= occurrence else None
