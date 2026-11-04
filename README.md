# Osky Remix Engine

Motor de producción de remixes headless, modular, no destructivo y reproducible, operado por
Claude desde una CLI, con interfaz web local para supervisión (etapa 6).

Estado: **etapa 0 completa**. Ver `docs/ROADMAP.md`.

## Requisitos del sistema (Ubuntu / Kubuntu)

```bash
./scripts/bootstrap.sh      # apt + Surge XT + uv sync + demucs
```

## Uso

```bash
uv run osky-remix doctor             # audita capacidades reales (verifica audio)
uv run osky-remix doctor --list      # pruebas disponibles
uv run osky-remix doctor -o surge -o mastering --keep   # algunas, conservando audios
```

## Pruebas

```bash
uv run pytest -m "not compat"   # unitarias, rápidas
uv run pytest -m compat         # capacidades reales (lentas)
```

## Documentación

- `docs/ARCHITECTURE.md` — separación de responsabilidades
- `docs/COMPATIBILITY.md` — resultados del doctor, tolerancias y hallazgos
- `docs/ROADMAP.md` — etapas y compuertas
- `AGENTS.md` — reglas para Claude al trabajar en este repo
