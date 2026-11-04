"""Paquete de entrega reproducible.

<salida>/
  audio/master.wav (24 bits)  audio/premaster.wav (float)  audio/stems/*.wav (alineados, float)
  midi/arrangement.mid        (un track por pista de instrumento, mapa de tempo exacto, PPQ 960)
  presets/<pista>.fxp         plugins.json (manifiesto)      quality.json / quality.md
  project/project.json        project/assets/<sha>.<ext>     (todo lo referenciado)
  manifest.json               (sha256 de cada archivo)        README.md
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import mido
import soundfile as sf

from ..core.models import MidiClip, Project, TrackKind
from .renderer import RENDER_VERSION, Renderer


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def write_midi(p: Project, path: Path) -> dict[str, int]:
    mf = mido.MidiFile(type=1, ticks_per_beat=p.ppq)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name=p.name, time=0))
    last = 0
    for tp in p.tempo:
        meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tp.bpm), time=tp.tick - last))
        last = tp.tick
    m0 = p.meter[0]
    meta.insert(1, mido.MetaMessage("time_signature", numerator=m0.numerator, denominator=m0.denominator, time=0))
    mf.tracks.append(meta)
    counts = {}
    for ch, t in enumerate(x for x in p.tracks if x.kind == TrackKind.instrument):
        ev = []
        drum = bool(t.plugins) and t.plugins[0].uri == "drums"
        chan = 9 if drum else ch % 9
        for c in t.clips:
            if isinstance(c, MidiClip):
                for n in c.notes:
                    ev.append((c.start_tick + n.tick, 1, n.pitch, n.velocity))
                    ev.append((c.start_tick + n.tick + n.duration, 0, n.pitch, 0))
        ev.sort(key=lambda e: (e[0], e[1]))      # note_off antes que note_on en el mismo tick
        tr = mido.MidiTrack()
        tr.append(mido.MetaMessage("track_name", name=t.id, time=0))
        now = 0
        for tick, on, pitch, vel in ev:
            tr.append(mido.Message("note_on" if on else "note_off", note=pitch, velocity=vel, channel=chan, time=tick - now))
            now = tick
        mf.tracks.append(tr)
        counts[t.id] = sum(1 for e in ev if e[1])
    mf.save(str(path))
    return counts


def plugin_manifest(p: Project, report) -> dict:
    import dawdreamer
    tracks = {}
    for t in p.tracks:
        tracks[t.id] = {
            "kind": t.kind.value, "output": t.output, "gain_db": t.gain_db, "pan": t.pan,
            "sidechain_from": t.sidechain_from, "sends": [s.model_dump() for s in t.sends],
            "plugins": [{"id": s.id, "backend": s.backend, "uri": s.uri, "preset_sha256": s.preset, "params": s.params,
                         "bypass": s.bypass} for s in t.plugins],
            "automation": [a.target for a in p.automation if a.track == t.id],
            "frozen_render_sha256": report.frozen.get(t.id),
        }
    surge = None
    try:
        import subprocess
        surge = subprocess.run(["dpkg-query", "-W", "-f=${Version}", "surge-xt"], capture_output=True, text=True).stdout or None
    except Exception:  # noqa: BLE001
        pass
    return {"render_version": RENDER_VERSION, "hosts": {"dawdreamer": dawdreamer.__version__, "surge_xt": surge},
            "mastering_profile": report.profile, "tracks": tracks}


def export_package(p: Project, assets, cache, out: Path, profile: str = "club", revision: str | None = None) -> dict:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} no está vacío")
    (out / "audio").mkdir(parents=True, exist_ok=True)
    renderer = Renderer(p, assets, cache)
    renderer.validate_capabilities()
    rep = renderer.render(out / "audio", profile, stems=True, name="mix")
    if not rep.mastering_converged or rep.quality["status"] == "FAIL":
        raise ValueError(f"entrega rechazada: mastering {rep.mastering_status}, calidad {rep.quality['status']}")
    a = out / "audio"
    (a / "mix_master.wav").rename(a / "master.wav")
    (a / "mix_premaster.wav").rename(a / "premaster.wav")
    (a / "mix_stems").rename(a / "stems")
    (a / "mix_report.json").rename(out / "render_report.json")
    (out / "midi").mkdir()
    notes = write_midi(p, out / "midi" / "arrangement.mid")
    (out / "presets").mkdir()
    for t in p.tracks:
        for s in t.plugins:
            if s.preset and assets.find(s.preset):
                shutil.copyfile(assets.find(s.preset), out / "presets" / f"{t.id}-{s.id}.fxp")
    (out / "plugins.json").write_text(json.dumps(plugin_manifest(p, rep), indent=2, ensure_ascii=False))
    (out / "quality.json").write_text(json.dumps(rep.quality, indent=2, ensure_ascii=False))
    q = rep.quality
    (out / "quality.md").write_text("# Reporte de calidad\n\n" + f"Estado: **{q['status']}** · {q['lufs']} LUFS · "
                                    f"{q['true_peak_dbtp']} dBTP · LRA ≈ {q['loudness_range_lu']} LU\n\n"
                                    "| Verificación | Estado | Detalle |\n|---|---|---|\n"
                                    + "\n".join(f"| {c['check']} | {c['status']} | {c['detail']} |" for c in q["checks"])
                                    + "\n\nAvisos del render:\n" + "\n".join(f"- {w}" for w in rep.warnings or ["(ninguno)"]) + "\n")
    pd = out / "project" / "assets"
    pd.mkdir(parents=True)
    (out / "project" / "project.json").write_text(json.dumps(json.loads(p.canonical_json()), indent=2, ensure_ascii=False))
    for sha, asset in p.assets.items():
        src = assets.find(sha)
        if src is not None:
            shutil.copyfile(src, pd / f"{sha}{asset.ext}")
    (out / "README.md").write_text(
        f"# {p.name}\n\nRevisión `{revision or '-'}` · perfil `{profile}` · motor de render {RENDER_VERSION}\n\n"
        "- `audio/master.wav`: master 24 bits. `audio/premaster.wav`: mezcla float sin masterizar.\n"
        "- `audio/stems/`: stems post-fader alineados (misma longitud, empiezan en 0).\n"
        "- `midi/arrangement.mid`: MIDI del arreglo (PPQ 960, mapa de tempo exacto).\n"
        "- `plugins.json`: plugins, presets (SHA-256), parámetros, automatización y renders congelados.\n"
        "- `project/`: proyecto canónico y todos sus assets por SHA-256.\n\n"
        "Verificar integridad: `osky-remix package verify <carpeta>`\n")
    files = sorted(f for f in out.rglob("*") if f.is_file())
    manifest = {"revision": revision, "files": {str(f.relative_to(out)): _sha(f) for f in files}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return {"report": rep, "midi_notes": notes, "files": len(files)}


def verify_package(root: Path) -> list[str]:
    """Lista de problemas (vacía = íntegro)."""
    problems = []
    m = json.loads((root / "manifest.json").read_text())
    for rel, sha in m["files"].items():
        f = root / rel
        if not f.exists():
            problems.append(f"falta {rel}")
        elif _sha(f) != sha:
            problems.append(f"alterado {rel}")
    listed = set(m["files"])
    extra = [str(f.relative_to(root)) for f in root.rglob("*") if f.is_file() and f.name != "manifest.json"
             and str(f.relative_to(root)) not in listed]
    problems += [f"no listado {e}" for e in extra]
    p = Project.model_validate_json((root / "project" / "project.json").read_text())
    for sha, a in p.assets.items():
        f = root / "project" / "assets" / f"{sha}{a.ext}"
        if not f.exists():
            problems.append(f"asset faltante {sha[:12]}")
        elif _sha(f) != sha:
            problems.append(f"asset alterado {sha[:12]}")
    stems = sorted((root / "audio" / "stems").glob("*.wav"))
    lens = {sf.info(str(s)).frames for s in stems} | {sf.info(str(root / "audio" / "premaster.wav")).frames}
    if len(lens) > 1:
        problems.append(f"stems desalineados: longitudes {sorted(lens)}")
    return problems
