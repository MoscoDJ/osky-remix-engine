"""Pruebas de capacidades. Cada prueba verifica el AUDIO resultante, no solo el código de salida.

Una prueba devuelve un dict con: status (PASS/FAIL/WARN/SKIP), metrics y detail.
Se ejecutan aisladas en un subproceso (ver runner.py): si un plugin revienta, se registra
el fallo y el resto de las pruebas sigue.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from . import fixtures as fx

SR = fx.SR


@dataclass(frozen=True)
class Check:
    name: str
    title: str
    critical: bool
    fn: Callable[[Path], dict]
    timeout_s: int = 300


def result(status: str, detail: str, **metrics) -> dict:
    return {"status": status, "detail": detail, "metrics": metrics}


def _rms_db(x: np.ndarray) -> float:
    r = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return 20 * np.log10(r) if r > 0 else -np.inf


def _snr_db(ref: np.ndarray, est: np.ndarray) -> float:
    err = ref - est
    return 10 * np.log10(np.sum(ref ** 2) / max(np.sum(err ** 2), 1e-20))


# --------------------------------------------------------------------------- python
PINNED = ["numpy", "scipy", "soundfile", "librosa", "pyloudnorm", "pedalboard", "dawdreamer",
          "pretty_midi", "mido", "basic_pitch", "onnxruntime", "essentia", "typer", "rich", "pydantic"]


def check_python(work: Path) -> dict:
    from importlib.metadata import version, PackageNotFoundError
    vers, missing = {}, []
    dist = {"basic_pitch": "basic-pitch", "pretty_midi": "pretty_midi"}
    for mod in PINNED:
        try:
            importlib.import_module(mod)
            try:
                vers[mod] = version(dist.get(mod, mod))
            except PackageNotFoundError:
                vers[mod] = "?"
        except Exception as e:  # noqa: BLE001
            missing.append(f"{mod}: {str(e)[:80]}")
    root = Path(__file__).resolve().parents[3]
    lock = root / "uv.lock"
    py_ok = sys.version_info[:2] == (3, 11)
    ok = py_ok and not missing and lock.exists()
    detail = f"Python {sys.version.split()[0]}, {len(vers)}/{len(PINNED)} paquetes, uv.lock {'presente' if lock.exists() else 'FALTA'}"
    if missing:
        detail += "; faltan: " + "; ".join(missing)
    return result("PASS" if ok else "FAIL", detail, python=sys.version.split()[0], packages=vers,
                  lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest()[:16] if lock.exists() else None)


# --------------------------------------------------------------------------- gpu
def _demucs_python() -> str | None:
    exe = shutil.which("demucs")
    if not exe:
        return None
    first = Path(exe).read_text(errors="ignore").splitlines()[0]
    return first[2:].strip() if first.startswith("#!") else None


GPU_PROBE = r"""
import json, time, torch
r = {"cuda": torch.cuda.is_available(), "torch": torch.__version__}
if r["cuda"]:
    p = torch.cuda.get_device_properties(0)
    r.update(name=p.name, vram_gb=round(p.total_memory / 2**30, 2), cc=f"{p.major}.{p.minor}")
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Conv1d(1, 32, 9, padding=4), torch.nn.ReLU(),
                              torch.nn.Conv1d(32, 32, 9, padding=4), torch.nn.ReLU(),
                              torch.nn.Conv1d(32, 1, 9, padding=4)).eval()
    x = torch.randn(4, 1, 44100)
    with torch.no_grad():
        ref = net(x)
        g = net.cuda(); xg = x.cuda(); g(xg); torch.cuda.synchronize()
        t = time.time(); out = g(xg); torch.cuda.synchronize(); r["gpu_ms"] = round((time.time() - t) * 1000, 2)
    r["max_abs_diff_vs_cpu"] = float((out.cpu() - ref).abs().max())
print(json.dumps(r))
"""


def check_gpu(work: Path) -> dict:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return result("WARN", "No hay nvidia-smi: se usará CPU para separación y modelos")
    q = subprocess.run([smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                       capture_output=True, text=True, timeout=30)
    py = _demucs_python()
    if not py:
        return result("FAIL", f"GPU visible ({q.stdout.strip()}) pero no hay entorno torch (demucs) para probarla")
    p = subprocess.run([py, "-c", GPU_PROBE], capture_output=True, text=True, timeout=240)
    if p.returncode != 0:
        return result("FAIL", "torch no pudo ejecutar el modelo: " + p.stderr.strip()[-300:], nvidia_smi=q.stdout.strip())
    r = json.loads(p.stdout.strip().splitlines()[-1])
    if not r.get("cuda"):
        return result("FAIL", "torch no ve CUDA", **r)
    ok = r["max_abs_diff_vs_cpu"] < 1e-3
    return result("PASS" if ok else "FAIL",
                  f"{r['name']} {r['vram_gb']} GB (cc {r['cc']}); conv1d 4x1s en {r['gpu_ms']} ms, "
                  f"dif. vs CPU {r['max_abs_diff_vs_cpu']:.1e}", driver=q.stdout.strip(), **r)


# --------------------------------------------------------------------------- demucs
def check_demucs(work: Path) -> dict:
    exe = shutil.which("demucs")
    if not exe:
        return result("FAIL", "demucs no está instalado (uv tool install demucs)")
    st = fx.band_mixture()
    src = work / "mixture.wav"
    sf.write(src, np.stack([st["mixture"]] * 2, 1), SR, subtype="PCM_24")
    py = _demucs_python()
    dev = "cpu"
    if py and subprocess.run([py, "-c", "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)"],
                             capture_output=True, timeout=120).returncode == 0:
        dev = "cuda"
    t = time.time()
    p = subprocess.run([exe, "-n", "htdemucs", "-d", dev, "-o", str(work / "sep"), str(src)],
                       capture_output=True, text=True, timeout=600)
    secs = time.time() - t
    if p.returncode != 0:
        return result("FAIL", "demucs falló: " + p.stderr.strip()[-300:])
    out = work / "sep" / "htdemucs" / "mixture"
    stems = {}
    for name in ("drums", "bass", "other", "vocals"):
        f = out / f"{name}.wav"
        if not f.exists():
            return result("FAIL", f"falta el stem {name}.wav")
        a, sr = sf.read(f)
        stems[name] = a.mean(1)
    n = len(st["mixture"])
    lens = {k: len(v) for k, v in stems.items()}
    recon = sum(stems.values())[:n]
    snr = _snr_db(st["mixture"], recon)
    # cada componente conocido debe estar mejor representado en su stem que en los demás
    corr = {k: float(np.corrcoef(st[k], stems[k][:n])[0, 1]) for k in ("drums", "bass", "other")}
    lvl = {k: round(_rms_db(v), 1) for k, v in stems.items()}
    ok_len = all(abs(l - n) <= 1 for l in lens.values())
    ok = ok_len and snr >= 15 and all(c >= 0.7 for c in corr.values())
    return result("PASS" if ok else "FAIL",
                  f"4 stems en {secs:.1f}s ({dev}); reconstrucción {snr:.1f} dB; "
                  f"correlación por stem " + ", ".join(f"{k} {v:.2f}" for k, v in corr.items()),
                  device=dev, seconds=round(secs, 1), reconstruction_snr_db=round(snr, 1),
                  stem_correlation=corr, stem_rms_db=lvl, lengths=lens, expected_len=n)


# --------------------------------------------------------------------------- surge xt
def _surge_render(state: Path | None, notes: list[tuple[int, float, float]], dur: float):
    import dawdreamer as daw
    from ..audio.plugin_host import SURGE_XT_VST3, PREROLL_S
    eng = daw.RenderEngine(SR, 512)
    s = eng.make_plugin_processor("surge", str(SURGE_XT_VST3))
    if state:
        # load_state devuelve None; la carga se verifica por el audio (ver check_surge)
        s.load_state(str(state))
    for m, t0, d in notes:
        s.add_midi_note(m, 100, PREROLL_S + t0, d)
    eng.load_graph([(s, [])])
    eng.render(PREROLL_S + dur)
    a = eng.get_audio()
    pre = a[:, : int(PREROLL_S * SR)]
    return a[:, int(PREROLL_S * SR):], float(np.abs(pre).max()), s, eng


def check_surge(work: Path) -> dict:
    from ..audio.plugin_host import SURGE_XT_VST3, SURGE_FACTORY, fxp_to_state_file
    if not SURGE_XT_VST3.exists():
        return result("FAIL", f"No existe {SURGE_XT_VST3}")
    # "EP 1" suena en la nota tocada; otros presets de fábrica transponen por diseño
    preset = SURGE_FACTORY / "Keys" / "EP 1.fxp"
    if not preset.exists():
        return result("FAIL", f"No existe el preset de referencia {preset}")
    state = fxp_to_state_file(preset, work / "preset.state")
    note = 57  # A3 = 220 Hz
    a_def, _, _, _ = _surge_render(None, [(note, 0.0, 1.5)], 2.0)
    a_pre, pre_peak, _, _ = _surge_render(state, [(note, 0.0, 1.5)], 2.0)
    peak = 20 * np.log10(np.abs(a_pre).max() + 1e-12)
    f0 = fx.f0_median(a_pre[:, : int(1.5 * SR)], SR, 60, 1000)
    err = fx.cents(f0, fx.midi_to_hz(note)) if np.isfinite(f0) else float("nan")
    # el preset debe sonar distinto al patch por defecto
    def spec(a):
        s = np.abs(np.fft.rfft(a.mean(0)[: SR]))
        return s / (np.linalg.norm(s) + 1e-12)
    sim = float(np.dot(spec(a_def), spec(a_pre)))
    sf.write(work / "surge_preset.wav", a_pre.T, SR)
    ok = peak > -40 and np.isfinite(err) and abs(err) < 30 and sim < 0.98
    return result("PASS" if ok else "FAIL",
                  f"preset '{preset.stem}' cargado; pico {peak:.1f} dBFS, f0 {f0:.1f} Hz ({err:+.0f} cents vs 220 Hz), "
                  f"similitud espectral con patch por defecto {sim:.2f}",
                  preset=str(preset), peak_dbfs=round(peak, 1), f0_hz=round(f0, 2), cents_error=round(err, 1),
                  spectral_similarity_vs_default=round(sim, 3), preroll_leak_peak=pre_peak)


# --------------------------------------------------------------------------- dawdreamer
def check_dawdreamer(work: Path) -> dict:
    import dawdreamer as daw
    from ..audio.plugin_host import SURGE_XT_VST3, SURGE_XT_FX_VST3, PREROLL_S, find_param
    from scipy.signal import butter, sosfilt
    notes = [(57, 0.0, 3.8)]
    # 1) automatización: volumen global de Surge de 0 a 1 en 4 s
    eng = daw.RenderEngine(SR, 512)
    s = eng.make_plugin_processor("surge", str(SURGE_XT_VST3))
    idx = find_param(s, "Global Volume")
    total = PREROLL_S + 4.0
    ramp = np.concatenate([np.zeros(int(PREROLL_S * SR)), np.linspace(0.0, 1.0, int(4.0 * SR))]).astype(np.float32)
    s.set_automation(idx, ramp)
    for m, t0, d in notes:
        s.add_midi_note(m, 100, PREROLL_S + t0, d)
    eng.load_graph([(s, [])])
    eng.render(total)
    a = eng.get_audio()[:, int(PREROLL_S * SR):]
    early, late = _rms_db(a[:, int(0.3 * SR):int(1.0 * SR)]), _rms_db(a[:, int(3.0 * SR):int(3.7 * SR)])
    auto_ok = late - early > 6
    # 2) guardar / cargar estado conserva parámetros. Se verifica por AUDIO: get_parameter()
    #    devuelve la caché del host y no refleja lo cargado con load_state.
    #    Se usa un preset determinista: el patch por defecto de Surge tiene aleatoriedad interna.
    from ..audio.plugin_host import fxp_to_state_file, DETERMINISTIC_PRESET
    base_state = fxp_to_state_file(DETERMINISTIC_PRESET, work / "base.state")

    def render_vol(value=None, state=None):
        e_ = daw.RenderEngine(SR, 512)
        p_ = e_.make_plugin_processor("s", str(SURGE_XT_VST3))
        p_.load_state(str(state or base_state))
        if value is not None:
            # set_parameter justo después de load_state lo pisa la carga asíncrona del patch;
            # la automatización constante se reafirma en cada bloque.
            p_.set_automation(idx, np.full(int(SR * (PREROLL_S + 1.5)), value, np.float32))
        p_.add_midi_note(57, 100, PREROLL_S, 1.0)
        e_.load_graph([(p_, [])])
        e_.render(PREROLL_S + 1.5)
        return e_, p_, e_.get_audio()
    _, _, a_def = render_vol()
    e_set, p_set, a_set = render_vol(0.37)
    p_set.save_state(str(work / "surge.state"))
    _, _, a_load = render_vol(state=work / "surge.state")
    state_diff = float(np.abs(a_set - a_load).max())
    vol_drop = _rms_db(a_def) - _rms_db(a_set)
    state_ok = state_diff < 1e-6 and vol_drop > 6
    restored = f"audio idéntico tras cargar (dif. {state_diff:.1e}), {vol_drop:.1f} dB bajo el default"
    # 3) plugin de efecto nativo procesando audio (Surge XT Effects)
    fx_ok, fx_detail = None, "sin Surge XT Effects"
    if SURGE_XT_FX_VST3.exists():
        eng3 = daw.RenderEngine(SR, 512)
        x = fx.band_mixture(4.0)["mixture"].astype(np.float32)
        pb = eng3.make_playback_processor("src", np.stack([x, x]))
        e = eng3.make_plugin_processor("sfx", str(SURGE_XT_FX_VST3))
        eng3.load_graph([(pb, []), (e, ["src"])])
        eng3.render(4.0)
        y = eng3.get_audio()
        fx_ok = np.isfinite(y).all() and _rms_db(y) > -60 and y.shape[1] >= int(3.9 * SR)
        fx_detail = f"Surge XT Effects procesó audio (RMS {_rms_db(y):.1f} dB, lat. {e.get_latency_samples()} muestras)"
    ok = auto_ok and state_ok and fx_ok is not False
    return result("PASS" if ok else "FAIL",
                  f"automatización Global Volume: {early:.1f} -> {late:.1f} dB; estado (Global Volume 0.37): {restored}; "
                  + fx_detail,
                  automation_delta_db=round(late - early, 1), state_max_diff=state_diff,
                  state_volume_drop_db=round(vol_drop, 1), effect_ok=fx_ok)


# --------------------------------------------------------------------------- rubberband
def check_rubberband(work: Path) -> dict:
    exe = shutil.which("rubberband")
    if not exe:
        return result("FAIL", "rubberband CLI no instalado (apt install rubberband-cli)")
    src = work / "tone.wav"
    sf.write(src, fx.harmonic_tone(440, 4.0, decay=0.1) * 0.5, SR)
    out_t, out_p = work / "tempo.wav", work / "pitch.wav"
    for args, out in ((["-T", "1.5"], out_t), (["-p", "3"], out_p)):
        p = subprocess.run([exe, "-q", "-F", "--fine", *args, str(src), str(out)], capture_output=True, text=True, timeout=120)
        if p.returncode != 0:
            return result("FAIL", "rubberband falló: " + p.stderr[-200:])
    yt, _ = sf.read(out_t); yp, _ = sf.read(out_p)
    dur_t, exp_t = len(yt) / SR, 4.0 / 1.5
    f_t, f_p = fx.f0_median(yt[: int(2 * SR)], SR), fx.f0_median(yp[: int(3 * SR)], SR)
    c_t, c_p = fx.cents(f_t, 440), fx.cents(f_p, 440 * 2 ** (3 / 12))
    ok = abs(dur_t - exp_t) / exp_t < 0.01 and abs(c_t) < 15 and abs(c_p) < 15 and abs(len(yp) / SR - 4.0) < 0.04
    return result("PASS" if ok else "FAIL",
                  f"tempo x1.5: {dur_t:.3f}s (esperado {exp_t:.3f}s), pitch {c_t:+.1f} cents; "
                  f"+3 st: {f_p:.1f} Hz ({c_p:+.1f} cents)",
                  tempo_duration_s=round(dur_t, 4), expected_s=round(exp_t, 4), tempo_pitch_cents=round(c_t, 1),
                  shift_f0_hz=round(f_p, 2), shift_cents=round(c_p, 1))


# --------------------------------------------------------------------------- basic pitch
def check_basic_pitch(work: Path) -> dict:
    import pretty_midi
    from ..analysis.transcribe import transcribe, _model_path
    src = work / "melody.wav"
    sf.write(src, fx.melody(), SR)
    mid_path = work / "melody.mid"
    t = time.time()
    clean, raw = transcribe(src, mid_path)
    secs = time.time() - t
    pm = pretty_midi.PrettyMIDI(str(mid_path))           # el MIDI escrito debe ser legible
    pred = [(n.pitch, n.start) for inst in pm.instruments for n in inst.notes]
    def hits_of(notes):
        return sum(any(p == m and abs(s - start) <= 0.07 for p, s in notes) for m, start, _ in fx.MELODY)
    hits, raw_hits = hits_of(pred), hits_of([(n.pitch, n.start) for n in raw])
    extra, raw_extra = len(pred) - hits, len(raw) - raw_hits
    ok = hits >= len(fx.MELODY) - 1 and extra <= 1
    return result("PASS" if ok else "FAIL",
                  f"{hits}/{len(fx.MELODY)} notas correctas (tono exacto, inicio ±70 ms), {extra} extra "
                  f"(crudo: {raw_hits}/{len(fx.MELODY)}, {raw_extra} extra antes de filtrar armónicos); "
                  f"MIDI legible ({len(pred)} notas) en {secs:.1f}s",
                  hits=hits, expected=len(fx.MELODY), extra_notes=extra, raw_hits=raw_hits,
                  raw_extra_notes=raw_extra, model=str(_model_path()))


# --------------------------------------------------------------------------- ffmpeg
def check_ffmpeg(work: Path) -> dict:
    ff, fp = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ff or not fp:
        return result("FAIL", "ffmpeg/ffprobe no instalados")
    rng = np.random.default_rng(5)
    x = (rng.standard_normal((SR * 3, 2)) * 0.2).clip(-1, 1)
    src = work / "src.wav"; sf.write(src, x, SR, subtype="PCM_24")
    flac, back, rs = work / "a.flac", work / "back.wav", work / "rs.wav"
    run = lambda *a: subprocess.run([ff, "-loglevel", "error", "-y", *a], capture_output=True, text=True, timeout=120)
    run("-i", str(src), "-metadata", "title=Prueba Osky", "-metadata", "artist=Doctor", str(flac))
    run("-i", str(flac), "-c:a", "pcm_s24le", str(back))
    run("-i", str(src), "-ar", "48000", str(rs))
    a, _ = sf.read(src, dtype="int32"); b, _ = sf.read(back, dtype="int32")
    exact = a.shape == b.shape and np.array_equal(a, b)
    meta = json.loads(subprocess.run([fp, "-v", "error", "-show_entries", "format_tags:stream=sample_rate,bits_per_raw_sample",
                                      "-of", "json", str(flac)], capture_output=True, text=True).stdout)
    tags = {k.lower(): v for k, v in meta.get("format", {}).get("tags", {}).items()}
    info = sf.info(rs)
    ok = exact and tags.get("title") == "Prueba Osky" and info.samplerate == 48000 and abs(info.duration - 3.0) < 0.005
    ver = subprocess.run([ff, "-version"], capture_output=True, text=True).stdout.split("\n")[0]
    return result("PASS" if ok else "FAIL",
                  f"WAV24→FLAC→WAV {'bit-exacto' if exact else 'NO exacto'}; metadatos leídos {tags}; "
                  f"remuestreo 48 kHz dur {info.duration:.4f}s",
                  version=ver, bit_exact=exact, tags=tags)


# --------------------------------------------------------------------------- mastering
def check_mastering(work: Path) -> dict:
    from ..mastering.loudness import integrated_lufs
    from ..mastering.true_peak import true_peak_dbtp, sample_peak_dbfs
    from ..mastering.dynamics import master, PROFILES
    # EBU Tech 3341, casos 1 y 2: seno 1 kHz estéreo a -23 y -33 dBFS -> -23 y -33 LUFS (±0.1)
    l23 = integrated_lufs(np.stack([fx.sine(1000, 20, amp=10 ** (-23 / 20))] * 2, 1), SR)
    l33 = integrated_lufs(np.stack([fx.sine(1000, 20, amp=10 ** (-33 / 20))] * 2, 1), SR)
    # pico entre muestras: seno a fs/4 con fase 45° -> muestras a -3.01 dBFS, true peak 0 dBTP
    s = fx.sine(SR / 4, 1.0, phase=np.pi / 4)
    sp, tp = sample_peak_dbfs(s), true_peak_dbtp(s, SR)
    # cadena de mastering sobre mezcla caliente
    mix = fx.band_mixture(10.0)["mixture"]
    hot = np.stack([mix, mix * 0.9], 1) * 1.8
    res = master(hot, SR, PROFILES["club"])
    clip = int(np.sum(np.abs(res.audio) >= 1.0))
    sf.write(work / "master.wav", res.audio, SR, subtype="PCM_24")
    ok = (abs(l23 + 23) <= 0.1 and abs(l33 + 33) <= 0.1 and abs(sp + 3.01) < 0.05 and abs(tp) < 0.2
          and res.true_peak_dbtp <= -1.0 + 1e-3 and abs(res.lufs + 9) <= 0.5 and clip == 0)
    return result("PASS" if ok else "FAIL",
                  f"EBU 3341: {l23:.2f}/{l33:.2f} LUFS (esperado -23/-33); fs/4: pico muestras {sp:.2f} dBFS, "
                  f"true peak {tp:.2f} dBTP; master 'club': {res.lufs:.2f} LUFS, {res.true_peak_dbtp:.2f} dBTP, "
                  f"{clip} muestras recortadas, {res.iterations} iteraciones",
                  ebu_case1=round(l23, 3), ebu_case2=round(l33, 3), isp_sample_peak=round(sp, 3), isp_true_peak=round(tp, 3),
                  master_lufs=round(res.lufs, 2), master_tp=round(res.true_peak_dbtp, 2), clipped_samples=clip, log=res.log)


# --------------------------------------------------------------------------- reproducibilidad
# Tolerancias documentadas (docs/COMPATIBILITY.md)
REPRO_TOL_DB = -120.0      # patches deterministas: diferencia máxima entre renders, dBFS
STOCH_TOL_LU = 0.1         # patches estocásticos: diferencia de sonoridad integrada, LU
STOCH_TOL_SPEC_DB = 1.0    # patches estocásticos: distancia espectral media (bandas de 1/3 oct.), dB

REPRO_SCRIPT = r"""
import sys, numpy as np
from pathlib import Path
from osky_remix.doctor.checks import _surge_render
from osky_remix.doctor import fixtures as fx
from osky_remix.mastering.dynamics import master, PROFILES
from osky_remix.audio.plugin_host import fxp_to_state_file
preset, out = Path(sys.argv[1]), Path(sys.argv[2])
st = fxp_to_state_file(preset, out.with_suffix(".state"))
a, *_ = _surge_render(st, [(57, 0.0, 0.4), (60, 0.5, 0.4), (64, 1.0, 0.4), (69, 1.5, 0.8)], 3.0)
mix = fx.band_mixture(3.0)["mixture"]
x = a.T[: len(mix)] + np.stack([mix, mix], 1) * 0.5
y = master(x, fx.SR, PROFILES["club"]).audio
np.save(out, y.astype(np.float32))
"""


def _third_octave_db(x: np.ndarray) -> np.ndarray:
    spec = np.abs(np.fft.rfft(x.mean(axis=1))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / SR)
    edges = 25 * 2 ** (np.arange(0, 30) / 3)
    return np.array([10 * np.log10(spec[(f >= lo) & (f < hi)].sum() + 1e-12) for lo, hi in zip(edges[:-1], edges[1:])])


def _two_renders(work: Path, preset: Path, tag: str):
    outs = []
    for i in (1, 2):
        o = work / f"{tag}{i}.npy"
        p = subprocess.run([sys.executable, "-c", REPRO_SCRIPT, str(preset), str(o)],
                           capture_output=True, text=True, timeout=240)
        if p.returncode != 0:
            raise RuntimeError(f"render {tag}{i} falló: " + p.stderr.strip()[-300:])
        outs.append(np.load(o).astype(np.float64))
    return outs


def check_reproducibility(work: Path) -> dict:
    from ..audio.plugin_host import DETERMINISTIC_PRESET, STOCHASTIC_PRESET
    from ..mastering.loudness import integrated_lufs
    # 1) cadena completa con patch determinista: debe ser bit-exacta entre procesos
    a, b = _two_renders(work, DETERMINISTIC_PRESET, "det")
    if a.shape != b.shape:
        return result("FAIL", f"longitudes distintas {a.shape} vs {b.shape}")
    diff = float(np.abs(a - b).max())
    diff_db = 20 * np.log10(diff) if diff > 0 else -np.inf
    det_ok = diff_db <= REPRO_TOL_DB and _rms_db(a) > -40
    # 2) patch estocástico: equivalencia perceptual
    c, d = _two_renders(work, STOCHASTIC_PRESET, "sto")
    s_diff = float(np.abs(c - d).max())
    lu = abs(integrated_lufs(c, SR) - integrated_lufs(d, SR))
    spec = float(np.mean(np.abs(_third_octave_db(c) - _third_octave_db(d))))
    sto_ok = lu <= STOCH_TOL_LU and spec <= STOCH_TOL_SPEC_DB
    h = [hashlib.sha256(x.astype(np.float32).tobytes()).hexdigest()[:16] for x in (a, b)]
    return result("PASS" if det_ok and sto_ok else "FAIL",
                  f"determinista ({DETERMINISTIC_PRESET.stem}): dif. máx "
                  f"{'0 (bit-exacto)' if diff == 0 else f'{diff_db:.1f} dBFS'} (tol. {REPRO_TOL_DB:.0f} dBFS); "
                  f"estocástico ({STOCHASTIC_PRESET.stem}): muestras difieren hasta {s_diff:.2f}, "
                  f"sonoridad {lu:.3f} LU (tol. {STOCH_TOL_LU}), espectro {spec:.2f} dB (tol. {STOCH_TOL_SPEC_DB}) "
                  f"-> se congela como asset",
                  deterministic_max_diff=diff, sha256=h, stochastic_max_diff=s_diff,
                  stochastic_lufs_diff=round(lu, 4), stochastic_spectral_diff_db=round(spec, 3))


# --------------------------------------------------------------------------- complementarios
def check_essentia(work: Path) -> dict:
    import essentia.standard as es
    st = fx.band_mixture(8.0, sr=44100)
    audio = (st["other"] + st["bass"]).astype(np.float32)
    key, scale, strength = es.KeyExtractor(profileType="edma")(audio)
    bpm = es.RhythmExtractor2013(method="multifeature")(st["mixture"].astype(np.float32))[0]
    ok = key == "D" and scale == "minor" and abs(bpm - 120) < 1.5
    return result("PASS" if ok else "WARN", f"tonalidad {key} {scale} ({strength:.2f}), esperado D minor; BPM {bpm:.1f} (esperado 120)",
                  key=f"{key} {scale}", bpm=round(float(bpm), 2))


def check_faust(work: Path) -> dict:
    import dawdreamer as daw
    eng = daw.RenderEngine(SR, 512)
    f = eng.make_faust_processor("faust")
    f.set_dsp_string('import("stdfaust.lib"); process = os.osc(330) * 0.5 <: _, _;')
    if not f.compile():
        return result("WARN", "Faust (vía DawDreamer) no compiló el DSP de prueba")
    eng.load_graph([(f, [])]); eng.render(1.0)
    y = eng.get_audio()
    f0 = fx.f0_median(y, SR)
    ok = abs(fx.cents(f0, 330)) < 10
    return result("PASS" if ok else "WARN", f"Faust embebido: oscilador 330 Hz -> {f0:.1f} Hz", f0_hz=round(f0, 2))


def check_csound(work: Path) -> dict:
    exe = shutil.which("csound")
    if not exe:
        return result("WARN", "csound no instalado (opcional)")
    csd = work / "t.csd"
    csd.write_text("""<CsoundSynthesizer>
<CsOptions>
-W -o %s
</CsOptions>
<CsInstruments>
sr=44100
ksmps=32
nchnls=1
0dbfs=1
instr 1
a1 oscili 0.5, 261.63
out a1
endin
</CsInstruments>
<CsScore>
i1 0 1
</CsScore>
</CsoundSynthesizer>
""" % (work / "cs.wav"))
    p = subprocess.run([exe, "-d", str(csd)], capture_output=True, text=True, timeout=60)
    if p.returncode != 0 or not (work / "cs.wav").exists():
        return result("WARN", "csound no generó audio: " + p.stderr[-200:])
    y, sr = sf.read(work / "cs.wav")
    f0 = fx.f0_median(y, sr)
    ok = abs(fx.cents(f0, 261.63)) < 10
    return result("PASS" if ok else "WARN", f"csound render C4 -> {f0:.1f} Hz", f0_hz=round(f0, 2))


CHECKS: list[Check] = [
    Check("python", "Python y dependencias", True, check_python, 120),
    Check("gpu", "GPU NVIDIA", False, check_gpu, 300),
    Check("demucs", "Demucs (separación)", True, check_demucs, 900),
    Check("surge", "Surge XT (preset + MIDI)", True, check_surge, 180),
    Check("dawdreamer", "DawDreamer (host, automatización, estado)", True, check_dawdreamer, 180),
    Check("rubberband", "Rubber Band", True, check_rubberband, 180),
    Check("basic_pitch", "Basic Pitch (transcripción)", True, check_basic_pitch, 300),
    Check("ffmpeg", "FFmpeg (conversión y metadatos)", True, check_ffmpeg, 120),
    Check("mastering", "Mastering (LUFS, true peak, limitador)", True, check_mastering, 180),
    Check("reproducibility", "Reproducibilidad de render", True, check_reproducibility, 600),
    Check("essentia", "Essentia (análisis complementario)", False, check_essentia, 180),
    Check("faust", "Faust (DSP programable)", False, check_faust, 120),
    Check("csound", "Csound (DSP programable)", False, check_csound, 120),
]
