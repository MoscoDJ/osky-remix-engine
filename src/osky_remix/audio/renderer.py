"""Renderizador: proyecto -> premaster float, master WAV 24 bits, stems y reporte de calidad.

Procesa el grafo DSP en orden topológico (``audio.graph``):
  pista = clips de audio + instrumento (Surge XT o integrado)
        + entradas (pistas que salen a ella, envíos pre/post-fader)
        -> efectos (con automatización por bloques) -> [pre-fader]
        -> ganancia y paneo (automatizables) -> [post-fader] -> salida
El master pasa por un perfil de mastering (gain staging, mono en graves, tilt, glue, limitador).

Congelado: cada render de Surge se guarda en la caché por clave de contenido (notas, preset,
parámetros, automatización, sr); con patches estocásticos re-renderizar da el mismo audio.
Calibración: la transposición propia de cada preset se mide y se compensa en el MIDI.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from ..core.assets import AssetStore
from ..core.cache import CacheStore, cache_key
from ..core.models import AudioClip, MidiClip, PluginSlot, Project, Track, TrackKind
from ..mastering.dynamics import MasterProfile, load_profile, master
from ..mastering.loudness import integrated_lufs
from ..mastering.quality import quality_report
from . import synth
from .automation import curve, lanes_for
from .effects import EFFECTS, apply_effect, pan_gain_curve
from .graph import build_graph, describe

RENDER_VERSION = "3.1.0"
INSTRUMENTS_BUILTIN = {"drums", "fx", "synth"}
TAIL_S = 4.0


@dataclass
class RenderReport:
    sample_rate: int
    length_samples: int
    lufs: float | None = None
    true_peak_dbtp: float | None = None
    premaster_lufs: float | None = None
    profile: dict = field(default_factory=dict)
    mastering_status: str = "unknown"
    mastering_converged: bool = False
    mastering_unmet: list[str] = field(default_factory=list)
    mastering_log: list[str] = field(default_factory=list)
    graph: list[str] = field(default_factory=list)
    tracks: dict[str, dict] = field(default_factory=dict)
    frozen: dict[str, str] = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)


class Renderer:
    def __init__(self, project: Project, assets: AssetStore, cache: CacheStore):
        self.p = project
        self.assets = assets
        self.cache = cache
        self.sr = project.sample_rate
        self.tm = project.tempo_map()
        self.bpm = project.tempo[0].bpm
        self.graph = build_graph(project)
        self.audible, self.feeders, self.solo_core = self._solo_routes()
        self.n = self._length()
        self.report = RenderReport(self.sr, self.n, graph=describe(project))
        self.pre: dict[str, np.ndarray] = {}
        self.post: dict[str, np.ndarray] = {}
        self.dry: dict[str, np.ndarray] = {}

    # ---------------------------------------------------------------- utilidades
    def _solo_routes(self) -> tuple[set[str], set[str], set[str]]:
        """Pistas audibles y pistas que solo alimentan envíos bajo el solo actual.

        - Audibles: las pistas en solo, todo lo que les llega por su *salida* (fuentes de un bus en
          solo) y todo lo que hay río abajo de ellas (salidas y envíos).
        - Alimentadoras: las que llegan a una pista en solo a través de algún *envío* (p. ej. la voz
          con envío al retorno en solo). Se procesan para que el envío suene, pero su salida
          directa no llega a ninguna pista audible: solo se escucha el retorno, como en un DAW.
        """
        ids = {t.id for t in self.p.tracks}
        selected = {t.id for t in self.p.tracks if t.solo}
        if not selected:
            return ids, set(), ids
        outputs = {t.id: t.output for t in self.p.tracks if t.output in ids}
        sends = {t.id: {s.target for s in t.sends if s.target in ids} for t in self.p.tracks}

        def closure(start: set[str], step) -> set[str]:
            seen, frontier = set(start), set(start)
            while frontier:
                frontier = {n for x in frontier for n in step(x)} - seen
                seen |= frontier
            return seen

        sources_of = lambda x: {tid for tid, out in outputs.items() if out == x}
        senders_of = lambda x: {tid for tid, dest in sends.items() if x in dest}
        downstream = lambda x: ({outputs[x]} if x in outputs else set()) | sends.get(x, set())
        core = closure(selected, sources_of)
        audible = closure(core, downstream)
        # Solo cuentan los envíos hacia el núcleo del solo: el envío de una pista fuera del solo a un
        # retorno que solo es audible por estar río abajo (p. ej. el bombo al reverb de la voz) calla.
        first = {x for c in core for x in senders_of(c)} - audible
        feeders = closure(first, lambda x: sources_of(x) | senders_of(x)) - audible
        return audible, feeders, core

    def _length(self) -> int:
        end = 0
        for t in self.p.tracks:
            for c in t.clips:
                s = self.tm.tick_to_sample(c.start_tick)
                end = max(end, s + self.p.audio_length(c) if isinstance(c, AudioClip)
                          else self.tm.tick_to_sample(c.start_tick + c.length_ticks))
        return end + int(TAIL_S * self.sr)

    def _asset_audio(self, sha: str, offset: int = 0, length: int = -1) -> np.ndarray:
        path = self.assets.find(sha)
        if path is None:
            raise FileNotFoundError(f"asset {sha[:12]} no está en el almacén")
        with sf.SoundFile(str(path)) as source:
            sr = source.samplerate
            source.seek(offset)
            x = source.read(length, dtype="float32", always_2d=True)
        if x.shape[1] == 1:
            x = np.repeat(x, 2, axis=1)
        if sr != self.sr:
            g = gcd(sr, self.sr)
            x = resample_poly(x, self.sr // g, sr // g, axis=0).astype(np.float32)
        return x[:, :2]

    def _notes_abs(self, t: Track) -> list[tuple[int, int, int, int]]:
        out = []
        for c in t.clips:
            if isinstance(c, MidiClip):
                for nt in c.notes:
                    a = self.tm.tick_to_sample(c.start_tick + nt.tick)
                    b = self.tm.tick_to_sample(c.start_tick + nt.tick + nt.duration)
                    out.append((a, b, nt.pitch, nt.velocity))
        return sorted(out)

    # ---------------------------------------------------------------- instrumentos
    def _render_builtin(self, uri: str, notes, params: dict) -> np.ndarray:
        y = np.zeros(self.n)
        for a, b, p, v in notes:
            if uri == "drums":
                h = synth.drum_hit(p, v, self.sr, a)
            elif uri == "fx":
                h = synth.fx_hit(p, (b - a) / self.sr, v, self.sr, a)
            else:
                h = synth.subtractive(440 * 2 ** ((p - 69) / 12), (b - a) / self.sr, v, self.sr, params.get("bright", 0.5))
            k = min(len(h), self.n - a)
            if k > 0:
                y[a:a + k] += h[:k]
        return np.stack([y, y], 1)

    def _preset_path(self, slot: PluginSlot) -> Path | None:
        if not slot.preset:
            return None
        p = self.assets.find(slot.preset)
        if p is None:
            raise FileNotFoundError(f"preset {slot.preset[:12]} no está en el almacén")
        return p

    def _calibrate(self, slot: PluginSlot) -> int:
        from ..doctor.fixtures import f0_median
        from .render_service import environment_manifest
        key = cache_key("surge_calibrate", "2", [slot.preset or "default"],
                        {"sr": self.sr, "params": slot.params, "environment": environment_manifest()})
        c = self.cache.get_json(key)
        if c is not None:
            return int(c["semitones"])
        y = self._surge_raw(slot, [(0, int(1.0 * self.sr), 60, 100)], int(1.5 * self.sr), slot.params, {})
        f0 = f0_median(y[: self.sr], self.sr, 25, 2500)
        semis = int(12 * round(np.log2(f0 / 261.6256))) if np.isfinite(f0) else 0
        if np.isfinite(f0) and abs(12 * np.log2(f0 / 261.6256) - semis) > 0.5:
            self.report.warnings.append(f"preset {slot.id}: transposición no múltiplo de octava "
                                        f"({12 * np.log2(f0 / 261.6256):+.1f} st); se compensa {semis:+d}")
        self.cache.put_json(key, {"semitones": semis, "f0": float(f0) if np.isfinite(f0) else None})
        return semis

    def _surge_raw(self, slot: PluginSlot, notes, n: int, params: dict, autom: dict[str, np.ndarray]) -> np.ndarray:
        import dawdreamer as daw
        from .plugin_host import PREROLL_S, SURGE_XT_VST3, find_param, fxp_to_state_file, set_static_params
        eng = daw.RenderEngine(self.sr, 512)
        s = eng.make_plugin_processor("inst", str(SURGE_XT_VST3))
        pre = self._preset_path(slot)
        if pre is not None:
            with tempfile.TemporaryDirectory() as td:
                s.load_state(str(fxp_to_state_file(pre, Path(td) / "p.state")))
        total = PREROLL_S + n / self.sr
        static = {k: v for k, v in params.items() if k not in autom}
        if static:
            set_static_params(s, static, total, self.sr)
        pre_n = int(PREROLL_S * self.sr)
        for name, arr in autom.items():
            full = np.concatenate([np.full(pre_n, arr[0]), arr]).astype(np.float32)
            s.set_automation(find_param(s, name), full)
        for a, b, p, v in notes:
            s.add_midi_note(int(p), int(v), PREROLL_S + a / self.sr, max(0.01, (b - a) / self.sr))
        eng.load_graph([(s, [])])
        eng.render(total)
        y = eng.get_audio().T[pre_n:]
        out = np.zeros((n, 2), np.float32)
        out[: min(n, len(y))] = y[:n, :2]
        return out

    def _render_surge(self, t: Track, slot: PluginSlot, notes) -> np.ndarray:
        semis = self._calibrate(slot)
        notes = [(a, b, p - semis, v) for a, b, p, v in notes if 0 <= p - semis <= 127]
        lanes = {tgt.split(".", 1)[1]: ln for tgt, ln in lanes_for(self.p, t.id).items() if tgt.startswith(slot.id + ".")}
        autom = {k: curve(ln, self.tm, self.n) for k, ln in lanes.items()}
        from .render_service import environment_manifest
        payload = {"environment": environment_manifest(), "notes": notes, "params": slot.params, "n": self.n, "sr": self.sr,
                   "automation": {k: ln.model_dump(mode="json") for k, ln in lanes.items()}}
        key = cache_key("surge_track", RENDER_VERSION, [slot.preset or "default"], payload)
        with self.cache.lock(cache_key("surge_generation", "1", [key])):
            hit = self.cache.get(key)
            if hit:
                y, _ = sf.read(str(hit / "audio.wav"), dtype="float32", always_2d=True)
            else:
                y = self._surge_raw(slot, notes, self.n, slot.params, autom)
                with tempfile.TemporaryDirectory() as td:
                    f = Path(td) / "audio.wav"
                    sf.write(f, y, self.sr, subtype="FLOAT")
                    self.cache.put_files(key, {"audio.wav": f}, {"track": t.id, "preset": slot.preset})
        self.report.frozen[t.id] = hashlib.sha256(y.tobytes()).hexdigest()
        return y

    # ---------------------------------------------------------------- pistas
    def _own(self, t: Track) -> np.ndarray:
        y = np.zeros((self.n, 2), np.float32)
        for c in t.clips:
            if isinstance(c, AudioClip):
                src = self._asset_audio(c.asset, c.source_offset, c.source_length)
                if c.fade_in:
                    k = min(round(c.fade_in * self.sr / (self.p.assets[c.asset].sample_rate or self.sr)), len(src))
                    src[:k] *= np.linspace(0, 1, k)[:, None]
                if c.fade_out:
                    k = min(round(c.fade_out * self.sr / (self.p.assets[c.asset].sample_rate or self.sr)), len(src))
                    if k:
                        src[len(src) - k:] *= np.linspace(1, 0, k)[:, None]
                a = self.tm.tick_to_sample(c.start_tick)
                k = min(len(src), self.n - a)
                y[a:a + k] += src[:k] * np.float32(10 ** (c.gain_db / 20))
        if t.kind == TrackKind.instrument and t.plugins:
            inst = t.plugins[0]
            notes = self._notes_abs(t)
            if notes and not inst.bypass:
                if inst.backend == "surge_xt":
                    y += self._render_surge(t, inst, notes)
                elif inst.backend == "builtin" and inst.uri in INSTRUMENTS_BUILTIN:
                    y += self._render_builtin(inst.uri, notes, inst.params).astype(np.float32)
                else:
                    raise ValueError(f"instrumento no soportado en {t.id}: {inst.backend}/{inst.uri}")
        return y

    def _effects(self, t: Track) -> list[PluginSlot]:
        chain = t.plugins[1:] if t.kind == TrackKind.instrument else t.plugins
        return [s for s in chain if not s.bypass]

    def _process(self, t: Track) -> None:
        x = self._own(t)
        for o in self.p.tracks:
            if o.id == t.id:
                continue
            # Una alimentadora del solo solo entrega su salida a otra alimentadora (cadena hacia
            # un envío); hacia una pista audible su señal seca queda en silencio.
            if o.output == t.id and o.kind != TrackKind.master and (o.id not in self.feeders or t.id in self.feeders):
                x = x + self.post[o.id]
            for s in o.sends:
                if s.target == t.id and (o.id not in self.feeders or t.id in self.feeders or t.id in self.solo_core):
                    src = self.pre[o.id] if s.pre_fader else self.post[o.id]
                    x = x + src * np.float32(10 ** (s.level_db / 20))
        self.dry[t.id] = x
        lanes = lanes_for(self.p, t.id)
        for slot in self._effects(t):
            if slot.backend != "builtin" or slot.uri not in EFFECTS:
                self.report.warnings.append(f"{t.id}: efecto {slot.backend}/{slot.uri} no soportado, omitido")
                continue
            autom = {tgt.split(".", 1)[1]: curve(ln, self.tm, self.n) for tgt, ln in lanes.items()
                     if tgt.startswith(slot.id + ".")}
            sc = self.dry.get(t.sidechain_from) if slot.uri == "sidechain" else None
            try:
                x = apply_effect(slot.uri, slot.params, x, self.sr, self.bpm, sc, autom).astype(np.float32)
            except ValueError as e:
                self.report.warnings.append(f"{t.id}: {e}; se aplica sin automatización")
                x = apply_effect(slot.uri, slot.params, x, self.sr, self.bpm, sc).astype(np.float32)
        if (t.id not in self.audible and t.id not in self.feeders) or t.mute:
            x = np.zeros_like(x)
        self.pre[t.id] = x
        g_db = curve(lanes["gain_db"], self.tm, self.n) if "gain_db" in lanes else np.full(self.n, t.gain_db)
        pan = curve(lanes["pan"], self.tm, self.n) if "pan" in lanes else np.full(self.n, t.pan)
        gl, gr = pan_gain_curve(pan)
        g = 10 ** (g_db / 20)
        x = (x * np.stack([g * gl, g * gr], 1)).astype(np.float32)
        if not np.isfinite(x).all():
            raise ValueError(f"la pista '{t.id}' produjo NaN/Inf")
        self.post[t.id] = x
        if t.kind != TrackKind.master:
            pk = float(np.abs(x).max())
            pre_pk = float(np.abs(self.pre[t.id]).max())
            self.report.tracks[t.id] = {"peak_dbfs": round(20 * np.log10(pk + 1e-12), 2),
                                        "rms_dbfs": round(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12), 2),
                                        "pre_fader_peak_dbfs": round(20 * np.log10(pre_pk + 1e-12), 2)}
            if pk < 1e-5 and (t.clips or t.kind in (TrackKind.bus, TrackKind.ret)):
                self.report.warnings.append(f"la pista '{t.id}' quedó en silencio")
            if pre_pk > 1.0:
                self.report.warnings.append(f"la pista '{t.id}' supera 0 dBFS antes del fader "
                                            f"({20 * np.log10(pre_pk):+.1f} dB): revisar gain staging")

    def output(self, tid: str) -> np.ndarray:
        if tid not in self.post:
            for x in self.graph.order:
                if x not in self.post:
                    self._process(self.p.track(x))
                if x == tid:
                    break
        return self.post[tid]

    def validate_capabilities(self):
        for t in self.p.tracks:
            if t.kind == TrackKind.instrument and t.plugins:
                inst = t.plugins[0]
                if not inst.bypass and not (inst.backend == "surge_xt" or
                   (inst.backend == "builtin" and inst.uri in INSTRUMENTS_BUILTIN)):
                    raise ValueError(f"instrumento no soportado: {t.id}/{inst.uri}")
            for slot in self._effects(t):
                if slot.backend != "builtin" or slot.uri not in EFFECTS:
                    raise ValueError(f"efecto no soportado para entrega: {t.id}/{slot.uri}")
            for slot in t.plugins:
                if slot.state and not slot.bypass:
                    raise ValueError(f"estado de plugin no soportado para entrega: {t.id}/{slot.id}")

    # ---------------------------------------------------------------- render completo
    def run(self, profile: str | MasterProfile | None = None):
        """Mezcla y master en memoria: (premaster, MasterResult, perfil)."""
        prof = profile if isinstance(profile, MasterProfile) else load_profile(profile or self.p.master_profile)
        pre = self.output(self.graph.master)
        measured = integrated_lufs(pre, self.sr)
        self.report.premaster_lufs = round(measured, 2) if np.isfinite(measured) else None
        res = master(pre, self.sr, prof)
        self.last_master = res.audio
        self.report.lufs = round(res.lufs, 2) if res.lufs is not None else None
        self.report.true_peak_dbtp = round(res.true_peak_dbtp, 2) if res.true_peak_dbtp is not None else None
        self.report.mastering_status = res.status
        self.report.mastering_converged = res.converged
        self.report.mastering_unmet = res.unmet
        self.report.profile = res.profile
        self.report.mastering_log = res.log
        if res.lufs is None:
            self.report.quality = {"status": "FAIL", "lufs": None, "true_peak_dbtp": res.true_peak_dbtp,
                                   "checks": [{"check": "loudness", "status": "FAIL", "detail": res.status}]}
        else:
            self.report.quality = quality_report(res.audio, self.sr, prof.ceiling_dbtp, prof.target_lufs, pre)
        if not res.converged:
            self.report.quality["status"] = "FAIL"
            self.report.warnings.append(f"mastering {res.status}: {', '.join(res.unmet)}")
        return pre, res, prof

    def render(self, out_dir: Path, profile: str | MasterProfile | None = None, stems: bool = True,
               name: str = "mix") -> RenderReport:
        out_dir.mkdir(parents=True, exist_ok=True)
        pre, res, prof = self.run(profile)
        files = {}
        f_pre = out_dir / f"{name}_premaster.wav"
        sf.write(f_pre, pre, self.sr, subtype="FLOAT")
        files["premaster"] = str(f_pre)
        f_m = out_dir / f"{name}_master.wav"
        sf.write(f_m, res.audio, self.sr, subtype="PCM_24")
        files["master"] = str(f_m)
        if stems:
            sd = out_dir / f"{name}_stems"
            sd.mkdir(exist_ok=True)
            for t in self.p.tracks:
                if t.kind != TrackKind.master and t.id in self.post:
                    sf.write(sd / f"{t.id}.wav", self.post[t.id], self.sr, subtype="FLOAT")
            files["stems"] = str(sd)
        self.report.files = files
        (out_dir / f"{name}_report.json").write_text(json.dumps(self.report.__dict__, indent=2, ensure_ascii=False))
        return self.report
