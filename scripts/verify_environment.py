"""Verifica los entornos externos sin instalarlos ni alterarlos."""
import json
import re
import subprocess
from osky_remix.core.config import REPO_ROOT
from osky_remix.separation.base import tool_python
from osky_remix.core.environment import separator_environment, unpinned_models


def _norm(name):
    return re.sub(r'[-_.]+', '-', name.lower())


def main():
    problems = []
    for backend, executable in [('demucs','demucs'), ('roformer','audio-separator')]:
        py = tool_python(executable)
        if not py and backend == "roformer":
            print(json.dumps({"tool":executable,"status":"SKIP","reason":"opcional no instalado"}))
            continue
        if not py:
            problems.append(f'{executable}: no instalado')
            continue
        environment = separator_environment(backend)
        actual = {_norm(k): v for k, v in environment['packages'].items()}
        name = 'demucs' if backend == 'demucs' else 'audio-separator'
        for line in (REPO_ROOT / f'requirements/{name}-linux.txt').read_text().splitlines():
            if '==' not in line:
                continue
            package, expected = (x.strip() for x in line.split('#', 1)[0].split('==', 1))
            installed = actual.get(_norm(package))
            if installed != expected:
                problems.append(f'{executable}: {package} requiere {expected}, instalado {installed or "ninguno"}')
        unpinned = unpinned_models(environment)
        print(json.dumps({'tool': executable, 'models': environment['models'], 'packages': len(actual),
                          **({'unpinned_models': unpinned} if unpinned else {})}))
        for name in unpinned:
            print(f'AVISO {executable}: modelo sin hash fijado en requirements/model-hashes.json: {name}')
    if problems:
        raise SystemExit('\n'.join(problems))


if __name__ == '__main__':
    main()
