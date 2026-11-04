"""Renderizador básico (etapa 3): proyecto -> premaster float, master WAV 24 bits y stems.

- Pistas de audio: clips desde assets (offset/longitud en muestras, ganancia, fades).
- Pistas de instrumento: el primer plugin es el instrumento (Surge XT o integrado);
  los demás son efectos integrados.
- Enrutado: salidas a buses/master, envíos a retornos, sidechain desde otra pista.
- Congelado: cada render de Surge se guarda en la caché por clave de contenido (notas,
  preset, parámetros, sr); con patches estocásticos re-renderizar da el mismo audio.
- Calibración: la transposición propia de cada preset se mide y se compensa en el MIDI.
La etapa 4 amplía esto (automatización, grafo DSP completo, A/B, exportación de paquete).
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
from ..mastering.dynamics import PROFILES, master
from ..mastering.loudness import integrated_lufs
from ..mastering.true_peak import true_peak_dbtp
from . import synth
from .effects import EFFECTS, apply_effect, pan_gain

RENDER_VERSION = "1.0.0"
INSTRUMENTS_BUILTIN = {"drums", "fx", "synth"}
TAIL_S = 4.0


@dataclass
class RenderReport:
    sample_rate: int
    length_samples: int
    lufs: float
    true_peak_dbtp: float
    premaster_lufs: float
    tracks: dict[str, dict] = field(default_factory=dict)
    frozen: dict[str, str] = field(default_factory=dict)
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
        self.report = RenderReport(self.sr, 0, 0, 0, 0)
        self.n = self._length()
        self.report.length_samples = self.n
        self._out: dict[str, np.ndarray] = {}
        self._dry: dict[str, np.ndarray] = {}
        self._busy: set[str] = set()

    # ---------------------------------------------------------------- utilidades
    def _length(self) -> int:
        end = 0
        for t in self.p.tracks:
            for c in t.clips:
                s = self.tm.tick_to_sample(c.start_tick)
                end = max(end, s + c.source_length if isinstance(c, AudioClip)
                          else self.tm.tick_to_sample(c.start_tick + c.length_ticks))
        return end + int(TAIL_S * self.sr)

    def _asset_audio(self, sha: str) -> np.ndarray:
        path = self.assets.find(sha)
        if path is None:
            raise FileNotFoundError(f"asset {sha[:12]} no está en el almacén")
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
        if x.shape[1] == 1:
            x = np.repeat(x, 2, axis=1)
        if sr != self.sr:
            g = gcd(sr, self.sr)
            x = resample_poly(x, self.sr // g, sr // g, axis=0).astype(np.float32)
        return x[:, :2]

    def _notes_abs(self, t: Track) -> list[tuple[int, int, int, int]]:
        """(muestra_inicio, muestra_fin, pitch, vel) de todos los clips MIDI de la pista."""
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
                f = 440 * 2 ** ((p - 69) / 12)
                h = synth.subtractive(f, (b - a) / self.sr, v, self.sr, params.get("bright", 0.5))
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
        """Semitonos que el preset transpone por diseño (múltiplo de 12), medido y cacheado."""
        from ..doctor.fixtures import f0_median
        key = cache_key("surge_calibrate", RENDER_VERSION, [slot.preset or "default"], {})
        c = self.cache.get_json(key)
        if c is not None:
            return int(c["semitones"])
        y = self._surge_raw(slot, [(0, int(1.0 * self.sr), 60, 100)], int(1.5 * self.sr), {})
        f0 = f0_median(y[: self.sr], self.sr, 25, 2500)
        semis = int(12 * round(np.log2(f0 / 261.6256))) if np.isfinite(f0) else 0
        if np.isfinite(f0) and abs(12 * np.log2(f0 / 261.6256) - semis) > 0.5:
            self.report.warnings.append(f"preset {slot.id}: transposición no múltiplo de octava "
                                        f"({12 * np.log2(f0 / 261.6256):+.1f} st); se compensa {semis:+d}")
        self.cache.put_json(key, {"semitones": semis, "f0": float(f0) if np.isfinite(f0) else None})
        return semis

    def _surge_raw(self, slot: PluginSlot, notes, n: int, params: dict) -> np.ndarray:
        import dawdreamer as daw
        from .plugin_host import PREROLL_S, SURGE_XT_VST3, fxp_to_state_file, set_static_params
        eng = daw.RenderEngine(self.sr, 512)
        s = eng.make_plugin_processor("inst", str(SURGE_XT_VST3))
        pre = self._preset_path(slot)
        if pre is not None:
            with tempfile.TemporaryDirectory() as td:
                st = fxp_to_state_file(pre, Path(td) / "p.state")
                s.load_state(str(st))
        total = PREROLL_S + n / self.sr
        if params:
            set_static_params(s, params, total, self.sr)
        for a, b, p, v in notes:
            s.add_midi_note(int(p), int(v), PREROLL_S + a / self.sr, max(0.01, (b - a) / self.sr))
        eng.load_graph([(s, [])])
        eng.render(total)
        y = eng.get_audio().T[int(PREROLL_S * self.sr):]
        out = np.zeros((n, 2), np.float32)
        out[: min(n, len(y))] = y[:n, :2]
        return out

    def _render_surge(self, t: Track, slot: PluginSlot, notes) -> np.ndarray:
        semis = self._calibrate(slot)
        notes = [(a, b, p - semis, v) for a, b, p, v in notes if 0 <= p - semis <= 127]
        payload = {"notes": notes, "params": slot.params, "n": self.n, "sr": self.sr}
        key = cache_key("surge_track", RENDER_VERSION, [slot.preset or "default"], payload)
        hit = self.cache.get(key)
        if hit:
            y, _ = sf.read(str(hit / "audio.wav"), dtype="float32", always_2d=True)
        else:
            y = self._surge_raw(slot, notes, self.n, slot.params)
            with tempfile.TemporaryDirectory() as td:
                f = Path(td) / "audio.wav"
                sf.write(f, y, self.sr, subtype="FLOAT")
                self.cache.put_files(key, {"audio.wav": f}, {"track": t.id, "preset": slot.preset})
        self.report.frozen[t.id] = hashlib.sha256(y.tobytes()).hexdigest()
        return y

    # ---------------------------------------------------------------- pistas
    def _own(self, t: Track) -> np.ndarray:
        """Audio propio de la pista (clips), antes de efectos."""
        y = np.zeros((self.n, 2), np.float32)
        for c in t.clips:
            if isinstance(c, AudioClip):
                src = self._asset_audio(c.asset)[c.source_offset:c.source_offset + c.source_length].copy()
                if c.fade_in:
                    src[: c.fade_in] *= np.linspace(0, 1, min(c.fade_in, len(src)))[:, None]
                if c.fade_out:
                    k = min(c.fade_out, len(src))
                    src[len(src) - k:] *= np.linspace(1, 0, k)[:, None]
                a = self.tm.tick_to_sample(c.start_tick)
                k = min(len(src), self.n - a)
                y[a:a + k] += src[:k] * np.float32(10 ** (c.gain_db / 20))
        if t.kind == TrackKind.instrument and t.plugins:
            inst = t.plugins[0]
            notes = self._notes_abs(t)
            if notes:
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

    def output(self, tid: str) -> np.ndarray:
        """Salida post-fader de una pista (memoizada)."""
        if tid in self._out:
            return self._out[tid]
        if tid in self._busy:
            raise ValueError(f"ciclo de enrutado en '{tid}'")
        self._busy.add(tid)
        t = self.p.track(tid)
        x = self._own(t)
        for o in self.p.tracks:                                   # entradas desde otras pistas
            if o.id != tid and o.output == tid and o.kind != TrackKind.master:
                x = x + self.output(o.id)
            for s in o.sends:
                if s.target == tid:
                    x = x + self.output(o.id) * np.float32(10 ** (s.level_db / 20))
        self._dry[tid] = x
        for slot in self._effects(t):
            if slot.backend != "builtin" or slot.uri not in EFFECTS:
                self.report.warnings.append(f"{tid}: efecto {slot.backend}/{slot.uri} no soportado, omitido")
                continue
            sc = self._dry_of(t.sidechain_from) if slot.uri == "sidechain" else None
            x = apply_effect(slot.uri, slot.params, x, self.sr, self.bpm, sc).astype(np.float32)
        if t.mute:
            x = np.zeros_like(x)
        gl, gr = pan_gain(t.pan)
        g = 10 ** (t.gain_db / 20)
        x = x * np.array([g * gl, g * gr], np.float32)
        if not np.isfinite(x).all():
            raise ValueError(f"la pista '{tid}' produjo NaN/Inf")
        self._busy.discard(tid)
        self._out[tid] = x
        if t.kind != TrackKind.master:
            pk = float(np.abs(x).max())
            self.report.tracks[tid] = {"peak_dbfs": round(20 * np.log10(pk + 1e-12), 2),
                                       "rms_dbfs": round(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12), 2)}
            if pk < 1e-5 and (t.clips or t.kind in (TrackKind.bus, TrackKind.ret)):
                self.report.warnings.append(f"la pista '{tid}' quedó en silencio")
        return x

    def _dry_of(self, tid: str | None) -> np.ndarray | None:
        if tid is None:
            return None
        self.output(tid)
        return self._dry[tid]

    # ---------------------------------------------------------------- render completo
    def render(self, out_dir: Path, profile: str = "club", stems: bool = True, name: str = "mix") -> RenderReport:
        out_dir.mkdir(parents=True, exist_ok=True)
        master_id = next(t.id for t in self.p.tracks if t.kind == TrackKind.master)
        pre = self.output(master_id)
        self.report.premaster_lufs = round(integrated_lufs(pre, self.sr), 2)
        files = {}
        f_pre = out_dir / f"{name}_premaster.wav"
        sf.write(f_pre, pre, self.sr, subtype="FLOAT")
        files["premaster"] = str(f_pre)
        res = master(pre, self.sr, PROFILES[profile])
        f_m = out_dir / f"{name}_master.wav"
        sf.write(f_m, res.audio, self.sr, subtype="PCM_24")
        files["master"] = str(f_m)
        self.report.lufs = round(res.lufs, 2)
        self.report.true_peak_dbtp = round(res.true_peak_dbtp, 2)
        if stems:
            sd = out_dir / f"{name}_stems"
            sd.mkdir(exist_ok=True)
            for t in self.p.tracks:
                if t.kind != TrackKind.master and t.id in self._out:
                    f = sd / f"{t.id}.wav"
                    sf.write(f, self._out[t.id], self.sr, subtype="FLOAT")
            files["stems"] = str(sd)
        self.report.files = files
        (out_dir / f"{name}_report.json").write_text(json.dumps(self.report.__dict__, indent=2, ensure_ascii=False))
        return self.report
