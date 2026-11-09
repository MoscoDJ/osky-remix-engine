"""Forma de canción con verdad conocida: canción sintética a 100 BPM con letra sincronizada."""
import numpy as np
import pytest

from osky_remix.composition import groove as G
from osky_remix.composition.arranger import load_template
from osky_remix.composition.context import ComposeContext, VocalSource
from osky_remix.composition.song_form import (CLIP_PRE_S, auto_form, generate_song_parts, parse_lrc,
                                              vocal_blocks)
from osky_remix.composition.theory import BAR, S16, chord_pcs

SR = 48000
BPM_SRC = 100.0
SPB = 60 / BPM_SRC


def synth_song():
    """Compases de 2.4 s. Estrofa (Dm) en los compases 4-11 con anacrusa de 1 tiempo; coro (D) en
    12-19 pegado a la estrofa; hueco de 4 compases; estrofa 24-27; coro 28-35. Frases de 2 compases
    con respiración de medio tiempo."""
    n_beats = 40 * 4 + 8
    beats = [int(round(i * SPB * SR)) for i in range(n_beats)]
    downs = beats[::4]
    env = np.full(int(n_beats * SPB * 100) + 10, -90.0)
    phrases = []                                       # (inicio, fin) en s

    def sing(bar0, bars, pickup=0.0):
        for k in range(0, bars, 2):
            a = (bar0 + k) * 4 * SPB - (pickup if k == 0 else 0) * SPB
            b = (bar0 + k + 2) * 4 * SPB - 0.5 * SPB
            env[int(a * 100):int(b * 100)] = -18
            phrases.append((a, b))
    sing(4, 8, pickup=1.0)
    sing(12, 8)
    sing(24, 4)
    sing(28, 8)
    lrc = "".join(f"[{int(a // 60):02d}:{a % 60:05.2f}]linea {i}\n" for i, (a, _) in enumerate(phrases))
    lrc += f"[{int((phrases[-1][1] + 1) // 60):02d}:{(phrases[-1][1] + 1) % 60:05.2f}]\n"
    keys = [(0, int(12 * 4 * SPB * SR), "D minor"), (int(12 * 4 * SPB * SR), int(24 * 4 * SPB * SR), "D major"),
            (int(24 * 4 * SPB * SR), int(28 * 4 * SPB * SR), "D minor"), (int(28 * 4 * SPB * SR), n_beats * int(SPB * SR), "D major")]
    chords = []
    for bar in range(40):
        lab = ["Dm", "Gm", "A", "Dm"][bar % 4] if not (12 <= bar < 24 or bar >= 28) else ["D", "G", "A", "D"][bar % 4]
        chords.append((downs[bar], downs[bar + 1] if bar + 1 < len(downs) else beats[-1], lab))
    bpm = 125.0
    ratio = bpm / BPM_SRC
    ctx = ComposeContext(bpm=bpm, key="D minor", source_bpm=BPM_SRC, beats=beats, b0=0, chords_src=chords,
                         downbeats=downs, sr=SR, vocal_env=env, key_segments=keys,
                         vocal=VocalSource("a" * 64, SR, 0, 60 / bpm * SR, int(n_beats * SPB * SR / ratio)))
    ctx.melody_provider = lambda beat, bars: [(0.0, 0.3, 69), (0.5, 0.8, 72), (1.0, 1.4, 74), (1.5, 1.7, 72), (2.0, 2.6, 69)]
    ctx.lyrics = parse_lrc(lrc)
    return ctx, phrases


def test_parse_lrc_keeps_empty_markers():
    ls = parse_lrc("[00:01.50]hola\n[01:02.25]\n[00:30.00]mundo")
    assert [(round(x.t, 2), x.text) for x in ls] == [(1.5, "hola"), (30.0, "mundo"), (62.25, "")]


def test_blocks_kinds_anchors_and_lengths():
    ctx, _ = synth_song()
    bl = vocal_blocks(ctx, ctx.lyrics)
    assert [b.kind for b in bl] == ["estrofa", "coro", "estrofa", "coro"]
    bar_of = lambda b: round(ctx.beats[b.anchor] / SR / (4 * SPB))
    assert [bar_of(b) for b in bl] == [4, 12, 24, 28]             # la anacrusa de 1 tiempo no mueve el ancla
    assert [b.bars for b in bl] == [8, 8, 4, 8]
    assert bl[0].onset_s == pytest.approx(4 * 4 * SPB - SPB, abs=0.02)


def test_auto_form_inserts_instrumentals_at_boundaries():
    ctx, _ = synth_song()
    tpl = load_template("melodic_techno_song")
    form = auto_form(vocal_blocks(ctx, ctx.lyrics), tpl["song_form"]["recipes"])
    assert [f["name"] for f in form] == ["intro", "estrofa1", "subida", "coro1", "drop", "break", "estrofa2",
                                         "coro2", "drop_final", "outro"]


def test_vocal_is_complete_cut_in_silence_and_bar_aligned():
    ctx, phrases = synth_song()
    tpl = load_template("melodic_techno_song")
    parts = generate_song_parts(tpl, ctx)
    vox = [c for t, c in parts.clips if t == "vox"]
    assert len(vox) == 4
    ratio = ctx.ratio
    # 1) toda frase cantada cae entera dentro de algún clip (nada se corta a media frase)
    for a, b in phrases:
        sa, sb = a * SR / ratio, b * SR / ratio
        assert any(c["source_offset"] <= sa and sb <= c["source_offset"] + c["source_length"] for c in vox), (a, b)
    # 2) cada clip empieza en silencio, 0.12 s antes del ataque
    for c in vox:
        t = c["source_offset"] * ratio / SR
        assert ctx.vocal_env[int(t * 100)] < -80
    # 3) el tiempo fuerte del original (ancla) cae en el inicio de un compás del remix
    tps = 125.0 / 60 * 960                              # ticks por segundo del remix
    starts = {s["name"]: sum(x["bars"] for x in parts.sections[:i]) * BAR for i, s in enumerate(parts.sections)}
    for c, name in zip(vox, ["estrofa1", "coro1", "estrofa2", "coro2"]):
        sec = next(s for s in parts.sections if s["name"] == name)
        anchor_t = ctx.beats[sec["_region"]["beat"]] / SR
        clip_t = c["source_offset"] * ratio / SR
        tick_anchor = c["start_tick"] + (anchor_t - clip_t) * BPM_SRC / 60 * 960
        assert abs(tick_anchor - starts[name]) <= 2
    # 4) bloques seguidos (estrofa2 → coro2): empalme en la misma muestra, sin fundidos
    e2, c2 = vox[2], vox[3]
    assert e2["source_offset"] + e2["source_length"] == pytest.approx(c2["source_offset"], abs=1)
    end_tick = e2["start_tick"] + e2["source_length"] / (60 / 125 * SR) * 960
    assert abs(end_tick - c2["start_tick"]) <= 2
    assert e2["fade_out"] == 0 and c2["fade_in"] == 0
    # el clip antes de un tramo instrumental se desvanece después del último sonido
    assert vox[0]["fade_out"] > 0


def test_automation_returns_to_mix_value_outside_its_section():
    ctx, _ = synth_song()
    tpl = load_template("melodic_techno_song")
    parts = generate_song_parts(tpl, ctx)
    lead = sorted((a for a in parts.autom if a["track"] == "lead" and a["param"] == "hz"), key=lambda a: a["t1"])
    drop_t = sum(s["bars"] for s in parts.sections[:[s["name"] for s in parts.sections].index("drop")]) * BAR
    at_drop = [a for a in lead if a["t1"] <= drop_t < a["t2"]]
    lp = next(x["params"]["hz"] for x in tpl["mix"]["lead"]["fx"] if x["uri"] == "lowpass")
    assert at_drop and at_drop[0]["from"] == at_drop[0]["to"] == lp


def test_groove_rules():
    hb = ["Dm", "Dm", "Gm", "Gm", "A", "A", "Dm", "Dm"]
    for style in ("light", "rolling", "rolling_fifth", "pulse"):
        for n in G.bass(hb, style):
            assert n.tick % (4 * S16) != 0                  # nunca sobre el bombo
    hats = [n for n in G.tops(2, 1, 0, swing=0.56) if n.pitch == G.CHH]
    off = {n.tick % S16 for n in hats if (n.tick // S16) % 2 == 1}
    assert off == {int(round(0.12 * S16))}                  # swing 56 % en las semicorcheas pares
    cell = G.hook_cell([(0.0, 0.3, 69), (0.31, 0.35, 70), (0.5, 0.8, 72), (1.0, 1.4, 74)])
    assert cell == [69, 72, 74]                             # sin el adorno de 40 ms
    notes = G.hook(hb * 2, cell, [2, 4, 5, 7, 9, 10, 0, 1], seed=1)
    for n in notes:
        if (n.tick % BAR) % (4 * S16) == 0:
            ch = (hb * 2)[n.tick // (2 * 960)]
            assert n.pitch % 12 in chord_pcs(ch)            # tiempo fuerte: nota del acorde
    assert 70 <= np.median([n.pitch for n in notes]) <= 88


def test_blocks_separated_by_silence_are_not_spliced():
    """Un adlib de un compás seguido de un hueco largo y otra estrofa (consecutivos en la forma):
    cada clip conserva su propia ancla; el segundo no se coloca con la escala del primero."""
    ctx, phrases = synth_song()
    env = ctx.vocal_env
    a = 0.5                                                  # adlib en el compás 0: 7.5 s antes de la estrofa
    env[int(a * 100):int((a + 1.0) * 100)] = -18
    ctx.lyrics = parse_lrc(f"[00:{a:05.2f}]adlib\n" + "".join(f"[{int(x // 60):02d}:{x % 60:05.2f}]linea {i}\n"
                                                               for i, (x, _) in enumerate(phrases)))
    tpl = load_template("melodic_techno_song")
    parts = generate_song_parts(tpl, ctx)
    names = [s["name"] for s in parts.sections]
    assert names[:3] == ["intro", "estrofa1", "estrofa2"]
    vox = [c for t, c in parts.clips if t == "vox"]
    starts = {s["name"]: sum(x["ticks"] for x in parts.sections[:i]) for i, s in enumerate(parts.sections)}
    sec = parts.sections[2]
    anchor_t = ctx.beats[sec["_region"]["beat"]] / SR
    clip_t = vox[1]["source_offset"] * ctx.ratio / SR
    assert abs(vox[1]["start_tick"] + (anchor_t - clip_t) * BPM_SRC / 60 * 960 - starts["estrofa2"]) <= 2
    assert vox[0]["fade_out"] > 0 and vox[1]["fade_in"] > 0


def test_irregular_bar_splits_block_into_pieces():
    """Un compás de 2 tiempos dentro del coro: el bloque se parte y la pieza siguiente arranca su
    rejilla en el tiempo fuerte real; la sección dura exactamente los tiempos del original."""
    ctx, _ = synth_song()
    # compás 15 de 2 tiempos: desde ahí todo se corre 2 tiempos
    beats = ctx.beats
    downs = [d for d in ctx.downbeats if d < beats[15 * 4]] + [beats[i] for i in range(15 * 4 + 2, len(beats), 4)]
    ctx.downbeats = downs
    bl = vocal_blocks(ctx, ctx.lyrics)
    coro = bl[1]
    assert [n for _, n in coro.pieces] == [14, coro.pieces[1][1]] and coro.pieces[1][0] == 15 * 4 + 2
    tpl = load_template("melodic_techno_song")
    parts = generate_song_parts(tpl, ctx)
    names = [s["name"] for s in parts.sections]
    assert "coro1b" in names
    c1 = parts.sections[names.index("coro1")]
    assert c1["ticks"] == 14 * 960
    kick = [c for t, c in parts.clips if t == "kick" and c["id"].endswith("coro1b")][0]
    assert kick["notes"][0]["tick"] == 0                    # la pieza reinicia el compás en su tiempo fuerte


def test_legato_phrase_before_instrumental_is_not_cut():
    """La estrofa liga con el coro (40 ms entre frases) y en el remix va una subida en medio:
    el clip de la estrofa conserva la frase hasta su último sonido."""
    ctx, phrases = synth_song()
    env = ctx.vocal_env
    end_e = 12 * 4 * SPB - 0.04                                  # la última frase de la estrofa llega al coro
    env[int((end_e - 1.0) * 100):int(end_e * 100)] = -18
    env[int(end_e * 100):int(end_e * 100) + 4] = -32             # caída entre sílabas (sin silencio)
    parts = generate_song_parts(load_template("melodic_techno_song"), ctx)
    vox = [c for t, c in parts.clips if t == "vox"]
    e1, c1 = vox[0], vox[1]
    end_clip = (e1["source_offset"] + e1["source_length"]) * ctx.ratio / SR
    assert end_clip >= end_e - 0.011
    start_c = c1["source_offset"] * ctx.ratio / SR
    assert start_c >= end_e - 0.03                               # el coro no arrastra la cola de la estrofa (±30 ms)
