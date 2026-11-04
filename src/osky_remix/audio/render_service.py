"""Identidad y publicación de renders compartidas por CLI y API."""
from dataclasses import asdict
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import json
import shutil
import tempfile
import os

from ..core.assets import sha256_file
from ..core.cache import cache_key, sweep_orphans, touch
from ..core.config import workspace
from ..mastering.dynamics import MasterProfile, load_profile
from .renderer import Renderer, RENDER_VERSION


def environment_manifest():
    packages = {}
    for name in ("numpy", "scipy", "soundfile", "pedalboard", "pyloudnorm", "dawdreamer"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    from .plugin_host import SURGE_XT_VST3
    root = Path(SURGE_XT_VST3)
    binaries = sorted(root.rglob("*.so")) if root.is_dir() else [root]
    return {"packages": packages, "surge": {str(p): sha256_file(p) for p in binaries if p.is_file()}}


def render_identity(revision, profile, environment=None):
    return cache_key("render", RENDER_VERSION, [revision],
                     {"profile": asdict(profile), "environment": environment if environment is not None else environment_manifest()})


def render_cached(repo, name, rev, profile=None):
    rid = repo.resolve(rev)
    project = repo.load(rid)
    prof = profile if isinstance(profile, MasterProfile) else load_profile(profile or project.master_profile)
    environment = environment_manifest()
    key = render_identity(rid, prof, environment)
    out = workspace() / "renders" / name / key
    required = ("mix_report.json", "mix_master.wav", "mix_premaster.wav", ".complete")
    def valid():
        if not all((out / f).is_file() for f in required):
            return False
        try:
            manifest = json.loads((out / ".complete").read_text())
            return all(sha256_file(out / f) == manifest[f] for f in required[:-1])
        except (ValueError, KeyError, OSError):
            return False
    with repo.cache.lock(key):
        if not valid():
            out.parent.mkdir(parents=True, exist_ok=True)
            sweep_orphans(out.parent, key)
            tmp = Path(tempfile.mkdtemp(prefix=key + "-", dir=out.parent))
            try:
                report = Renderer(project, repo.assets, repo.cache).render(tmp, prof, stems=False)
                report.files = {k: str(out / Path(v).name) for k, v in report.files.items()}
                data = {**report.__dict__, "environment": environment, "revision": rid, "cache_key": key}
                (tmp / "mix_report.json").write_text(json.dumps(data, allow_nan=False, indent=2))
                (tmp / ".complete").write_text(json.dumps({f: sha256_file(tmp / f) for f in required[:-1]}))
                if out.exists():
                    shutil.rmtree(out)
                os.replace(tmp, out)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        touch(out)
        return rid, json.loads((out / "mix_report.json").read_text()), out
