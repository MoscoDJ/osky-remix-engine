"""Forma de canción: la letra completa en orden y la música que toma el poder donde no hay voz.

1. **Bloques de voz** (``vocal_blocks``): las líneas de la letra (LRC) se agrupan en bloques; un
   bloque nuevo empieza tras un hueco largo sin voz (``BLOCK_GAP_S``) o cuando cambia el modo
   (p. ej. Re menor en las estrofas → Re mayor en el coro). Tipo: ``coro`` si su modo difiere del
   del primer bloque, ``estrofa`` si no.
2. **Cortes en silencio**: cada bloque empieza 0.12 s antes del primer ataque de voz (medido en el
   stem cerca de la marca del LRC) y termina tras el último sonido de voz del bloque más una cola;
   nunca a media frase. El ancla es el tiempo fuerte del original que corresponde al primer compás
   del bloque en el remix (la anacrusa, si la hay, cae antes); un compás del original = un compás
   del remix, así que la armonía leída del original queda debajo de la voz.
3. **Forma automática** (``auto_form``): intro → [estrofas] → subida (antes del primer coro) →
   coro → drop + break (tras el primer coro) → … → coro final → drop final → outro. Los tramos
   instrumentales repiten la armonía de 8 compases del bloque al que pertenecen.
4. **Partes** (``generate_song_parts``): recetas por tipo de tramo (plantilla ``song_form``):
   bombo, tops, bajo, pad, arpegio, gancho, loops de percusión, FX y automatización. En los
   bloques de voz, el gancho solo responde en compases donde la voz calla.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np

from ..core.models import MidiNote
from . import groove as G
from . import harmony as H
from .context import ComposeContext
from .theory import BAR, TICKS_PER_BEAT, key_scale

BLOCK_GAP_S = 6.0          # sin voz durante más que esto: bloque nuevo
CLIP_PRE_S = 0.12          # el clip empieza antes del ataque (consonante/respiración)
CLIP_TAIL_S = 0.6          # cola tras el último sonido de voz del bloque
VOICE_ON_DB, VOICE_OFF_DB = -30.0, -40.0


@dataclass
class LyricLine:
    t: float
    text: str


@dataclass
class VocalBlock:
    kind: str                  # "estrofa" | "coro"
    lines: list[int]           # índices en la lista de líneas con texto
    onset_s: float             # primer ataque de voz (original)
    offset_s: float            # último sonido de voz (original)
    anchor: int                # índice de beat (tiempo fuerte) del original = compás 0 del bloque
    bars: int
    mode: str
    gaps: list[int] = field(default_factory=list)      # compases (del bloque) sin voz
    pieces: list[tuple[int, int]] = field(default_factory=list)   # (beat inicial, tiempos): partido en compases irregulares
    free: bool = False         # cantado en tempo libre (sin rejilla): va sin batería
    end_beat: int = -1         # beat (del original) donde termina el bloque


def parse_lrc(text: str) -> list[LyricLine]:
    out = []
    for m in re.finditer(r"\[(\d+):(\d+(?:\.\d+)?)\]([^\n\r]*)", text):
        out.append(LyricLine(int(m[1]) * 60 + float(m[2]), m[3].strip()))
    return sorted(out, key=lambda x: x.t)


def _env_at(env: np.ndarray, t: float) -> float:
    i = int(t * 100)
    return float(env[i]) if 0 <= i < len(env) else -120.0


def _onset_near(env: np.ndarray, t: float) -> float:
    """Ataque de voz más cercano a la marca del LRC (que indica el inicio de la línea)."""
    a, b = max(0, int((t - 1.2) * 100)), min(len(env), int((t + 0.6) * 100))
    best = None
    for i in range(a, b):
        if env[i] > VOICE_ON_DB and env[max(0, i - 25):i].max(initial=-120) < VOICE_OFF_DB:
            if best is None or abs(i / 100 - t) < abs(best - t):
                best = i / 100
    if best is not None:
        return best
    # sin silencio previo (la frase anterior se liga): si ya hay voz 0.3 s antes de la marca, la
    # frontera es la caída de energía entre sílabas más cercana a la marca; si no, el primer ataque
    a2, b2 = max(0, int((t - 0.3) * 100)), min(len(env), int((t + 0.15) * 100))
    if a2 < b2 and env[a2] > VOICE_ON_DB:
        return (a2 + int(np.argmin(env[a2:b2]))) / 100
    for i in range(a2, b):
        if env[i] > VOICE_ON_DB:
            return i / 100
    return t


def _last_voice(env: np.ndarray, t_from: float, t_to: float) -> float:
    a, b = int(t_from * 100), min(len(env), int(t_to * 100))
    idx = np.flatnonzero(env[a:b] > VOICE_OFF_DB)
    return (a + idx[-1]) / 100 if len(idx) else t_from


def _mode_at(ctx: ComposeContext, t0: float, t1: float) -> str:
    """Modo mayoritario en [t0, t1] según los tramos de tonalidad del análisis."""
    segs = getattr(ctx, "key_segments", None) or []
    votes: dict[str, float] = {}
    for a, b, key in segs:
        a, b = a / ctx.sr, b / ctx.sr
        ov = min(b, t1) - max(a, t0)
        if ov > 0:
            votes[key] = votes.get(key, 0.0) + ov
    return max(votes, key=votes.get) if votes else ctx.key


def _anchor(ctx: ComposeContext, onset: float) -> int:
    """Tiempo fuerte del compás 0 del bloque: el siguiente si el ataque es una anacrusa de hasta
    1.5 tiempos (o 0.25 tiempos tarde), si no el anterior al ataque."""
    beats = np.asarray(ctx.beats) / ctx.sr
    downs = np.asarray(ctx.downbeats) / ctx.sr
    per = ctx.period
    nxt = downs[downs >= onset - 0.25 * per]
    if len(nxt) and nxt[0] - onset <= 1.5 * per:
        d = nxt[0]
    else:
        prev = downs[downs <= onset]
        d = prev[-1] if len(prev) else downs[0]
    return int(np.argmin(np.abs(beats - d)))


def vocal_blocks(ctx: ComposeContext, lyrics: list[LyricLine]) -> list[VocalBlock]:
    env = ctx.vocal_env
    if env is None:
        raise ValueError("la forma de canción necesita la voz (--vocals) para medir ataques")
    lines = [ln for ln in lyrics if ln.text]
    if not lines:
        raise ValueError("la letra no tiene líneas con texto")
    ends = [ln.t for ln in lyrics[1:]] + [lyrics[-1].t + 12]
    nxt_t = {id(ln): e for ln, e in zip(lyrics, ends)}
    onsets = [_onset_near(env, ln.t) for ln in lines]
    # último sonido de cada línea: hasta la marca siguiente, pero nunca más allá del ataque de la
    # línea siguiente (si el cantante liga, la frase anterior termina donde empieza la otra)
    limits = [min(nxt_t[id(ln)] + 0.4, (onsets[i + 1] - 0.02) if i + 1 < len(lines) else 1e9)
              for i, ln in enumerate(lines)]
    offsets = [_last_voice(env, on, lim) for on, lim in zip(onsets, limits)]
    modes = [_mode_at(ctx, on, max(on + 1, (nxt_t[id(ln)] + on) / 2 + 2)) for on, ln in zip(onsets, lines)]
    beats = np.asarray(ctx.beats) / ctx.sr
    free_until = beats[0] if len(beats) else 0.0                  # sin rejilla antes del primer beat
    is_free = [on < free_until - 0.5 for on in onsets]
    # sin cambio de modo en la canción: el coro se reconoce por las líneas que se repiten
    norm = lambda t: unicodedata.normalize("NFD", t.lower()).encode("ascii", "ignore").decode().strip(" ,.!¡¿?")
    counts: dict[str, int] = {}
    for ln in lines:
        counts[norm(ln.text)] = counts.get(norm(ln.text), 0) + 1
    rep = [counts[norm(ln.text)] > 1 for ln in lines]
    by_mode = len(set(modes)) > 1
    label = modes if by_mode else ["coro" if r else "estrofa" for r in rep]
    if not by_mode:
        # suavizado: una línea suelta distinta de sus vecinas (o la última, distinta de la anterior)
        # toma el tipo de la anterior
        for i in range(1, len(label)):
            nxt = label[i + 1] if i + 1 < len(label) else label[i - 1]
            if label[i] != label[i - 1] and nxt != label[i]:
                label[i] = label[i - 1]
    groups, cur = [], [0]
    for i in range(1, len(lines)):
        gap = onsets[i] - offsets[i - 1]
        if gap > BLOCK_GAP_S or label[i] != label[cur[0]] or is_free[i] != is_free[cur[0]]:
            groups.append(cur)
            cur = [i]
        else:
            cur.append(i)
    groups.append(cur)
    first = next((g for g in groups if not is_free[g[0]]), groups[0])
    blocks = []
    for g in groups:
        on, off = onsets[g[0]], max(offsets[i] for i in g)
        kind = ("coro" if label[g[0]] != label[first[0]] else "estrofa") if by_mode else label[g[0]]
        blocks.append(VocalBlock(kind="libre" if is_free[g[0]] else kind, lines=g, onset_s=on, offset_s=off,
                                 anchor=0 if is_free[g[0]] else _anchor(ctx, on), bars=0, mode=modes[g[0]],
                                 free=is_free[g[0]]))
    down_idx = sorted({int(np.argmin(np.abs(beats - d / ctx.sr))) for d in ctx.downbeats})
    for i, b in enumerate(blocks):
        if b.free:
            # sin rejilla: la duración en tiempos del remix sale de la duración real
            dur_beats = (b.offset_s + CLIP_TAIL_S - b.onset_s + CLIP_PRE_S) * ctx.source_bpm / 60
            b.bars = max(1, int(np.ceil(dur_beats / 4)))
            b.pieces = [(0, b.bars * 4)]
            continue
        # fin: tiempo fuerte tras el último sonido (la cola puede pasar medio tiempo al tramo
        # siguiente); nunca más allá del ancla del bloque siguiente
        end_t = b.offset_s - 0.5 * ctx.period
        downs = [d for d in ctx.downbeats if d / ctx.sr >= end_t]
        end_beat = int(np.argmin(np.abs(beats - downs[0] / ctx.sr))) if downs else len(beats) - 1
        nxt_grid = next((x for x in blocks[i + 1:] if not x.free), None)
        if nxt_grid is not None:
            end_beat = min(end_beat, nxt_grid.anchor)
        nbeats = end_beat - b.anchor
        b.end_beat = end_beat
        b.bars = max(1, int(round(nbeats / 4)))
        # piezas: tramos de compases regulares; un compás irregular (p. ej. de 2 tiempos) se suma
        # a la pieza anterior y la siguiente reinicia la rejilla del remix en su tiempo fuerte
        cuts = [d for d in down_idx if b.anchor < d < end_beat and (d - b.anchor) % 4 != 0]
        starts, a = [b.anchor], b.anchor
        for d in down_idx:
            if a < d < end_beat and (d - a) % 4 != 0:
                starts.append(d)
                a = d
        bounds = starts + [end_beat]
        b.pieces = [(x, y - x) for x, y in zip(bounds, bounds[1:]) if y > x]
        del cuts
        # compases del bloque sin voz (para que el gancho responda)
        for k in range(b.bars):
            t0 = beats[min(b.anchor + 4 * k, len(beats) - 1)]
            t1 = beats[min(b.anchor + 4 * k + 4, len(beats) - 1)]
            seg = env[int(t0 * 100):int(t1 * 100)]
            if len(seg) and seg.max() < VOICE_OFF_DB and t0 > b.onset_s:
                b.gaps.append(k)
    return blocks


def auto_form(blocks: list[VocalBlock], recipes: dict) -> list[dict]:
    """Secuencia de tramos: dicts con ``type`` (receta), ``name`` y ``block``/``piece`` o ``bars``.

    Un bloque partido en piezas (compases irregulares) ocupa entradas seguidas; los tramos
    instrumentales solo se insertan antes de su primera pieza o después de la última. Un bloque
    cantado en tempo libre (``libre``) abre la canción sin batería, antes del intro.
    """
    grid = [i for i, b in enumerate(blocks) if not b.free]
    first_grid = grid[0] if grid else 0

    def inst(kind: str, name: str, harm_block: int) -> dict:
        hb = harm_block if not blocks[harm_block].free else first_grid
        return {"type": kind, "name": name, "bars": recipes[kind]["bars"], "harmony_block": hb}

    def pieces(i: int, typ: str, name: str) -> list[dict]:
        n = len(blocks[i].pieces) or 1
        return [{"type": typ, "name": name if j == 0 else f"{name}{chr(ord('b') + j - 1)}", "block": i, "piece": j}
                for j in range(n)]
    coros = [i for i, b in enumerate(blocks) if b.kind == "coro"]
    first_c, last_c = (coros[0], coros[-1]) if coros else (None, None)
    form = []
    for i, b in enumerate(blocks):
        if b.free and i < first_grid:
            form += pieces(i, "libre", f"apertura{i + 1}" if i else "apertura")
    form.append(inst("intro", "intro", first_grid))
    after_break = False
    n_est, n_coro = 0, 0
    for i, b in enumerate(blocks):
        if b.free and i < first_grid:
            continue
        if b.kind == "coro":
            if i == first_c and not after_break and i > first_grid:
                form.append(inst("subida", "subida", i - 1))
            n_coro += 1
            form += pieces(i, "coro", f"coro{n_coro}")
            if i == last_c:
                form += [inst("drop_final", "drop_final", i), inst("outro", "outro", i)]
            elif i == first_c:
                form += [inst("drop", "drop", i), inst("break", "break", i)]
                after_break = True
        else:
            n_est += 1
            form += pieces(i, "regreso" if after_break else "estrofa", f"estrofa{n_est}")
    if last_c is None:                                  # canción sin coro detectado
        form += [inst("drop_final", "drop_final", grid[-1] if grid else 0), inst("outro", "outro", grid[-1] if grid else 0)]
    return form


def _halfbars_loop(ctx: ComposeContext, b0: int, loop_bars: int, bars: int) -> list[str]:
    loop = ctx.halfbars(0, loop_bars, b0=b0)
    return [loop[i % len(loop)] for i in range(bars * 2)]


def _clip(cid: str, start: int, length: int, notes: list[MidiNote]) -> dict:
    return {"type": "midi", "id": cid, "start_tick": start, "length_ticks": length,
            "notes": [n.model_dump() for n in sorted(notes, key=lambda x: (x.tick, x.pitch)) if n.tick < length]}


def _from(notes: list[MidiNote], bar: int) -> list[MidiNote]:
    return [n for n in notes if n.tick >= bar * BAR]


@dataclass
class SongParts:
    clips: list[tuple[str, dict]]
    autom: list[dict]
    sections: list[dict]
    used: set[str]
    blocks: list[VocalBlock]
    cell: list[int]
    tempo: list[tuple[int, float]] = field(default_factory=list)      # (tick, bpm) por tramo


def generate_song_parts(tpl: dict, ctx: ComposeContext, prefix: str = "", variant: str = "A") -> SongParts:
    sf = tpl["song_form"]
    recipes = sf["recipes"]
    lyrics = getattr(ctx, "lyrics", None)
    if not lyrics:
        raise ValueError("la forma de canción necesita la letra sincronizada (--lyrics archivo.lrc)")
    vs = ctx.vocal
    if vs is None:
        raise ValueError("la forma de canción necesita la voz (--vocals)")
    blocks = vocal_blocks(ctx, lyrics)
    form = auto_form(blocks, recipes)
    inst = tpl["instruments"]
    seed0 = ctx.seed * 100 + {"A": 0, "B": 11, "C": 23}.get(variant, 0)
    # gancho: contorno de la primera frase del primer coro (o de la primera estrofa)
    src = next((b for b in blocks if b.kind == "coro"), blocks[0])
    cell = G.hook_cell(ctx.melody_for(src.anchor, 4)) if ctx.melody_provider else []
    clips: list[tuple[str, dict]] = []
    autom: list[dict] = []
    sections: list[dict] = []
    t0 = 0
    ratio = ctx.ratio
    last_vox = None
    tempo: list[tuple[int, float]] = []
    for si, f in enumerate(form):
        r = recipes[f["type"]]
        name = f["name"]
        seed = seed0 + si
        piece = f.get("piece", 0)
        if "block" in f:
            b = blocks[f["block"]]
            pa, pbeats = b.pieces[piece] if b.pieces else (b.anchor, b.bars * 4)
            nb = int(np.ceil(pbeats / 4))
            if b.free:
                ref = blocks[next((j for j, x in enumerate(blocks) if not x.free), 0)]
                first = ctx.halfbars(0, 1, b0=ref.anchor)[0] if not ref.free else "N"
                hb = [first] * (2 * nb)                 # bordón del primer acorde
            else:
                hb = ctx.halfbars(0, nb, b0=pa)
            mode = b.mode
        else:
            nb = f["bars"]
            hb_block = blocks[f["harmony_block"]]
            hb = _halfbars_loop(ctx, hb_block.anchor, min(8, hb_block.bars), nb)
            mode = hb_block.mode
            b = None
        sec_bpm = float(r.get("bpm") or ctx.bpm)
        if b is not None and b.free:
            # sin rejilla: compases suficientes para la duración real de la voz (ya estirada) a este tempo
            dur_s = (b.offset_s + CLIP_TAIL_S - b.onset_s + CLIP_PRE_S) / ratio
            nb = int(np.ceil((dur_s * sec_bpm / 60 + 0.5) / 4))
            pbeats = nb * 4
            hb = (hb * (2 * nb))[: 2 * nb]
        L = nb * BAR if "block" not in f else pbeats * TICKS_PER_BEAT
        scale = key_scale(mode)
        if b is not None and not b.free and abs(sec_bpm - ctx.bpm) > 1e-6:
            raise ValueError(f"{name}: un bloque de voz va al tempo de la voz ({ctx.bpm}), no a {sec_bpm}")
        if not tempo or abs(tempo[-1][1] - sec_bpm) > 1e-6:
            tempo.append((t0, sec_bpm))

        def put(track: str, notes: list[MidiNote], tag: str = ""):
            if notes:
                clips.append((track, _clip(f"{prefix}{track}-{name}{tag}", t0, L, notes)))

        style = r.get("drums", sf.get("drums", "four"))
        if style == "four":
            if r.get("kick"):
                k = G.kicks(nb, drop_phrase_end=r["kick"] == "phrase_drop")
                put("kick", _from(k, r.get("kick_from_bar", 0)))
            if r.get("tops"):
                put("drums", _from(G.tops(nb, r["tops"], seed, sf.get("swing", 0.56), r.get("fills", True)),
                                   r.get("tops_from_bar", 0)))
        elif r.get("kick") or r.get("tops"):
            fn = {"halfstep": G.halfstep, "dembow": G.dembow}[style]
            pat = _from(fn(nb, max(1, r.get("tops", 1)), seed, r.get("fills", True)), r.get("tops_from_bar", 0))
            if r.get("kick"):
                put("kick", _from([n for n in pat if n.pitch == G.KICK], r.get("kick_from_bar", 0)))
            put("drums", [n for n in pat if n.pitch != G.KICK])
        if r.get("snare_build"):
            nbld = min(nb, r["snare_build"])
            put("drums", [n.model_copy(update={"tick": n.tick + (nb - nbld) * BAR}) for n in G.snare_build(nbld)], "-build")
        if r.get("bass"):
            lo, hi = inst["bass"].get("range", [33, 45])
            if r["bass"] in ("sustained", "wobble"):
                bn = G.bass_sustained(hb, lo, hi)
            elif r["bass"] == "808":
                bn = G.bass_808(hb, lo, hi)
            else:
                bn = G.bass(hb, r["bass"], lo, hi, seed)
            put("bass", _from(bn, r.get("bass_from_bar", 0)))
        if r.get("sub") and "sub" in inst:
            lo, hi = inst["sub"].get("range", [26, 38])
            put("sub", _from(G.bass_sustained(hb, lo, hi), r.get("bass_from_bar", 0)))
        if r.get("wobble"):
            w = r["wobble"]
            for a, b2, v1, v2 in G.wobble_points(nb, w.get("rates", [2, 4]), w.get("lo", 250), w.get("hi", 2500), seed):
                autom.append({"track": w.get("track", "bass"), "fx": w.get("fx", "lowpass"), "param": "hz",
                              "from": v1, "to": v2, "t1": t0 + a, "t2": t0 + b2})
        if r.get("pad"):
            lo, hi = inst["pad"].get("range", [50, 74])
            put("pad", H.pads(hb, lo, hi))
        if r.get("arp"):
            lo, hi = inst["arp"].get("range", [62, 86])
            put("arp", _from(H.arp(hb, seed, lo, hi, r.get("arp_shape", "pulse")), r.get("arp_from_bar", 0)))
        reg_hi = inst["lead"].get("register", 79)
        hk = r.get("hook")
        if hk in ("full", "teaser", "soft"):
            notes = G.hook(hb, cell, scale, seed0, reg_hi, variation=si)
            if hk == "teaser":
                notes = _from(notes, r.get("hook_from_bar", nb // 2))
            if hk == "soft":
                notes = [n.model_copy(update={"velocity": max(1, n.velocity - 24)}) for n in notes]
            put("lead", notes)
        elif hk == "answers" and b is not None and b.gaps and piece == 0:
            ans = []
            for g in b.gaps:
                a = G.hook_answer(hb[2 * g: 2 * g + 4] or hb[:4], cell, scale, seed0, reg_hi)
                ans += [n.model_copy(update={"tick": n.tick + g * BAR}) for n in a]
            put("lead", ans)
        # loops de percusión (audio, a tempo del proyecto)
        for role in r.get("loops", []):
            lp = (getattr(ctx, "loops", None) or {}).get(role)
            if not lp or abs(lp.get("bpm", sec_bpm) - sec_bpm) / sec_bpm > 0.002:
                continue                                 # un loop solo suena a su tempo
            step = int(lp["beats"]) * TICKS_PER_BEAT
            start_bar = r.get("loops_from_bar", 0)
            for j, tick in enumerate(range(start_bar * BAR, L, step)):
                length = min(lp["frames"], int(round((L - tick) / TICKS_PER_BEAT * 60 / sec_bpm * lp["sr"])))
                clips.append(("perc", {"type": "audio", "id": f"{prefix}perc-{role}-{name}-{j}", "asset": lp["sha"],
                                       "start_tick": t0 + tick, "source_offset": 0, "source_length": length,
                                       "fade_in": 0, "fade_out": int(0.004 * lp["sr"]), "gain_db": lp.get("gain_db", 0.0)}))
        fxn = []
        for fx in r.get("fx", []):
            if fx == "riser_end" and nb >= 4:
                fxn.append(MidiNote(tick=(nb - 4) * BAR, duration=4 * BAR, pitch=1, velocity=110))
            elif fx == "impact_start":
                fxn.append(MidiNote(tick=0, duration=BAR, pitch=2, velocity=110))
            elif fx == "downlifter_start":
                fxn.append(MidiNote(tick=0, duration=4 * BAR, pitch=3, velocity=100))
        put("fx", fxn)
        # voz: el bloque completo, cortado en silencios
        region = None
        if b is not None and piece == 0:
            t_a = ctx.beats[b.anchor] / ctx.sr
            nxt = next((blocks[j] for j in range(f["block"] + 1, len(blocks))), None)
            prev_f = form[si - 1] if si > 0 else {}
            last_piece = max(j for j, x in enumerate(form) if x.get("block") == f["block"])
            next_f = form[last_piece + 1] if last_piece + 1 < len(form) else {}
            # empalme en la misma muestra solo si en el original los bloques son contiguos (el
            # anterior termina en el ancla de este) y van seguidos en el remix
            joined_prev = (prev_f.get("block") == f["block"] - 1 and not b.free
                           and not blocks[f["block"] - 1].free and blocks[f["block"] - 1].end_beat == b.anchor)
            joined_next = (next_f.get("block") == f["block"] + 1 and not b.free
                           and not blocks[f["block"] + 1].free and b.end_beat == blocks[f["block"] + 1].anchor)
            prv = blocks[f["block"] - 1] if f["block"] > 0 else None
            cs = b.onset_s - CLIP_PRE_S
            if prv is not None and not joined_prev:
                cs = max(cs, prv.offset_s + 0.01)              # sin la cola de la frase anterior
            ce = b.offset_s + CLIP_TAIL_S
            if nxt is not None:
                # seguido en el remix: corte donde empieza el clip siguiente (sin pérdida, sin
                # duplicar); separado por un tramo instrumental: hasta justo antes de su ataque,
                # para no cortar una frase ligada a la siguiente
                ce = min(ce, nxt.onset_s - CLIP_PRE_S) if joined_next else min(ce, nxt.onset_s - 0.01)
            # bloques seguidos en el remix: corte en la misma muestra del original, sin fundidos
            fade_in = 0 if joined_prev else int(0.008 * vs.sr)
            fade_out = 0 if joined_next else int(np.clip(ce - b.offset_s, 0.05, 0.3) * vs.sr)
            k_tick = ctx.source_bpm / 60 * TICKS_PER_BEAT
            if joined_prev and last_vox is not None:
                # misma escala que el clip anterior: el empalme cae en el mismo instante del original
                start_tick = last_vox[1] + int(round((cs - last_vox[0]) * k_tick))
            elif b.free:
                start_tick = t0 + TICKS_PER_BEAT // 2          # sin rejilla: medio tiempo de aire al inicio
            else:
                start_tick = t0 + int(round((cs - t_a) * k_tick))
            last_vox = (cs, start_tick)
            off = int(round(cs * ctx.sr / ratio))
            length = int(round((ce - cs) * ctx.sr / ratio))
            if start_tick < 0 or off < 0:
                raise ValueError(f"{name}: la anacrusa del bloque cae antes del inicio del remix")
            length = min(length, vs.frames - off)
            clips.append(("vox", {"type": "audio", "id": f"{prefix}vox-{name}", "asset": vs.asset,
                                  "start_tick": start_tick, "source_offset": off, "source_length": length,
                                  "fade_in": fade_in, "fade_out": fade_out}))
            region = {"beat": b.anchor, "pickup_beats": round((t_a - cs) * ctx.source_bpm / 60, 2) if not b.free else 0.0,
                      "label": f"{b.kind}: líneas {b.lines[0] + 1}-{b.lines[-1] + 1}",
                      "onset_s": round(b.onset_s, 2), "offset_s": round(b.offset_s, 2)}
        for a in r.get("automation", []):
            fb = a.get("from_bar", 0)
            tb = a.get("to_bar", nb)
            fb, tb = (nb + fb if fb < 0 else fb), (nb + tb if tb < 0 else tb)
            autom.append({**{k: v for k, v in a.items() if k not in ("from_bar", "to_bar")},
                          "t1": t0 + fb * BAR, "t2": t0 + tb * BAR})
        sections.append({"name": name, "type": f["type"], "bars": nb, "ticks": L, "energy": r.get("energy", 2),
                         "vocal": b is not None, "_t": (t0, t0 + L), **({"_region": region} if region else {})})
        t0 += L
    # un parámetro automatizado en algún tramo vuelve a su valor de la mezcla en los demás (si no,
    # el carril se quedaría en el último valor: p. ej. el filtro del gancho cerrado en el drop)
    keys = {(a["track"], a["fx"], a["param"]) for a in autom}
    for tr, fx, param in keys:
        default = next((x["params"].get(param) for x in tpl.get("mix", {}).get(tr, {}).get("fx", [])
                        if x["uri"] == fx and param in x.get("params", {})), None)
        if default is None:
            continue
        for s in sections:
            a0, a1 = s["_t"]
            if not any(a["track"] == tr and a["fx"] == fx and a["param"] == param and a0 <= a["t1"] < a1 for a in autom):
                autom.append({"track": tr, "fx": fx, "param": param, "from": default, "to": default, "t1": a0, "t2": a1})
    for s in sections:
        s.pop("_t")
    # orden por inicio: en un tick compartido, el final del tramo anterior queda antes que el
    # inicio del siguiente (el carril salta al valor nuevo, no se queda en el viejo)
    autom.sort(key=lambda a: a["t1"])
    return SongParts(clips=clips, autom=autom, sections=sections, used={t for t, _ in clips}, blocks=blocks, cell=cell,
                     tempo=tempo)
