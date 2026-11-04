"""Huella de herramientas externas y modelos realmente disponibles."""
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from .assets import sha256_file
from .config import REPO_ROOT


@lru_cache(maxsize=None)
def _sha256_cached(path: str, size: int, mtime_ns: int) -> str:
    # Indexado por tamaño y fecha: un checkpoint de cientos de MB se lee una sola vez por proceso,
    # y si se reemplaza en disco la clave cambia y se vuelve a calcular.
    return sha256_file(Path(path))


def _sha256(p: Path) -> str:
    st = p.stat()
    return _sha256_cached(str(p), st.st_size, st.st_mtime_ns)


@lru_cache(maxsize=None)
def _packages(py: str) -> dict:
    # Se asume estable durante la vida del proceso (un comando CLI o un trabajo aislado).
    result = subprocess.run([py, "-c", "import json;from importlib.metadata import distributions;print(json.dumps({d.metadata['Name']:d.version for d in distributions()}))"],
                            check=True, capture_output=True, text=True, timeout=60)
    return json.loads(result.stdout)


def model_pins() -> dict:
    return json.loads((REPO_ROOT / "requirements/model-hashes.json").read_text())


def _model_paths(backend):
    if backend == "roformer":
        from ..separation.roformer import MODEL_DIR
        return [p for p in MODEL_DIR.glob("*") if p.suffix in (".ckpt", ".yaml", ".onnx")]
    root = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache/torch")) / "hub/checkpoints"
    paths = list(root.glob("*.th"))
    hfroot = Path(os.environ.get("HF_HUB_CACHE", Path(os.environ.get("HF_HOME", Path.home()/".cache/huggingface"))/"hub"))
    for pattern in ("models--adefossez--HTDemucs*", "models--adefossez--Demucs*"):
        for modelroot in hfroot.glob(pattern):
            ref = modelroot / "refs/main"
            if ref.is_file():
                revision = ref.read_text().strip()
                paths.extend(p for p in (modelroot / "snapshots" / revision).glob("*")
                             if p.suffix in (".yaml", ".safetensors"))
    return paths


def _model_name(backend, p):
    if backend != "demucs":
        return p.name
    return "demucs/" + p.parts[-4] + "/" + p.name if "snapshots" in p.parts else "demucs/legacy/" + p.name


def separator_environment(backend):
    """Paquetes y hashes de los modelos del separador. Falla si un modelo fijado fue alterado.

    Se puede llamar varias veces en un mismo proceso: los hashes y la lista de paquetes se
    reutilizan mientras los archivos no cambien.
    """
    from ..separation.base import tool_python
    executable = "demucs" if backend == "demucs" else "audio-separator"
    py = tool_python(executable)
    packages = dict(_packages(py)) if py else {}
    pins = model_pins()
    models = {_model_name(backend, p): _sha256(p) for p in sorted(_model_paths(backend))}
    for name, digest in models.items():
        if name in pins and pins[name] != digest:
            raise ValueError(f"modelo alterado: {name}")
    return {"packages": packages, "models": models}


def unpinned_models(environment: dict) -> list[str]:
    """Modelos presentes sin hash fijado: funcionan, pero su reproducibilidad no está garantizada."""
    pins = model_pins()
    return sorted(name for name in environment["models"] if name not in pins)
