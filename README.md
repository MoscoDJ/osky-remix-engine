# Osky Remix Engine

Motor de producción de remixes headless, modular, no destructivo y reproducible, operado por
Claude desde una CLI, con interfaz web local para supervisión (etapa 6).

Estado: **etapas 0 y 1 completas**. Ver `docs/ROADMAP.md`.

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

## Proyectos y operaciones

```bash
uv run osky-remix project new mi-remix --bpm 124 --sr 48000
uv run osky-remix asset import mi-remix cancion.flac --kind original
uv run osky-remix op apply mi-remix ops.json -a claude -m "Arreglo base"   # lote atómico
echo '[{"op":"lock_track","track":"vox"}]' | uv run osky-remix op apply mi-remix - -a user -m "Voz final"
uv run osky-remix project branch mi-remix arreglo-b --from main
uv run osky-remix project log mi-remix
uv run osky-remix project merge mi-remix arreglo-b -a claude
uv run osky-remix project revert mi-remix <rev> -a user
```

El contrato está en `docs/PROJECT_SCHEMA.md` y `schemas/`.

## Pruebas

```bash
uv run pytest -m "not compat"   # unitarias, rápidas
uv run pytest -m compat         # capacidades reales (lentas)
```

## Documentación

- `docs/ARCHITECTURE.md` — separación de responsabilidades
- `docs/PROJECT_SCHEMA.md` — modelo, unidades, operaciones y revisiones
- `docs/COMPATIBILITY.md` — resultados del doctor, tolerancias y hallazgos
- `docs/ROADMAP.md` — etapas y compuertas
- `AGENTS.md` — reglas para Claude al trabajar en este repo
