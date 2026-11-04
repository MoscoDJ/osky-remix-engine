# Osky Remix Engine

Motor de producción de remixes headless, modular, no destructivo y reproducible, operado por
Claude desde una CLI, con interfaz web local para supervisión (etapa 6).

Estado: **etapas 0 a 6 completas**; hitos 1 (32 compases de melodic techno con voz extraída) y 2 (tres arreglos sin perder ediciones manuales) cumplidos. Ver `docs/ROADMAP.md`.

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

## Interfaz web

```bash
(cd frontend && npm ci && npm run build)
uv run osky-remix serve        # http://127.0.0.1:8765
```

Timeline, mezclador, editor MIDI, render, A/B, evaluación e historial. Ver `docs/UI.md`.

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

## Análisis y separación

```bash
uv run osky-remix ingest mi-remix cancion.flac                  # original + copia de trabajo
uv run osky-remix separate mi-remix <sha> -b demucs --lead-backing
uv run osky-remix analyze mi-remix <sha> --stems <sha-reporte> --md analisis.md
uv run osky-remix reference mi-remix cancion.mid --analysis <sha> --stems <sha-reporte>   # MIDI de referencia
```

Métodos, confianza y verificación en `docs/ANALYSIS.md`.

## Producción

```bash
uv run osky-remix project branch mi-remix idea --from main
uv run osky-remix compose mi-remix --analysis <sha> --vocals <sha-voz> --at 1:23.1 --bars 32 --branch idea
uv run osky-remix render mi-remix --rev idea
uv run osky-remix transpose mi-remix <sha-voz> --to-key proyecto --analysis <sha> --clip <clip>   # voz a la tonalidad
```

Mezcla, A/B y entrega:

```bash
uv run osky-remix graph mi-remix --rev idea
uv run osky-remix compare mi-remix main idea
uv run osky-remix export mi-remix entrega/ --rev idea && uv run osky-remix package verify entrega/
```

Ver `docs/AUDIO_ENGINE.md`.

Los renders se guardan por revisión, perfil y entorno, así que se acumulan. Para limpiar lo
que lleva tiempo sin usarse (todo es regenerable; los proyectos no se tocan):

```bash
uv run osky-remix gc --older-than 30            # informa qué borraría
uv run osky-remix gc --older-than 30 --apply    # borra; --cache incluye la caché de proyectos
```

## Agente

```bash
uv run osky-remix evaluate mi-remix --rev idea
uv run osky-remix improve mi-remix --branch idea
uv run osky-remix vary mi-remix --from idea --variants A,B,C
```

`vary` solo regenera contenido generado sin editar. Los proyectos creados antes del registro
de procedencia (`generated_content`) se conservan enteros salvo con `--adopt-legacy`.

Ver `docs/AGENT.md` y `docs/CODEX_WORKFLOW.md`.

## Pruebas

```bash
uv run pytest -m "not compat and not audio"   # unitarias, rápidas
uv run pytest -m audio          # análisis sobre fixtures sintéticos (~1 min)
uv run pytest -m compat         # capacidades reales (lentas)
uv run pytest -m ui             # interfaz con Playwright (requiere frontend compilado)
```

## Documentación

- `docs/ARCHITECTURE.md` — separación de responsabilidades
- `docs/UI.md` — interfaz web local
- `docs/AGENT.md` — evaluador, acciones, planificador, variaciones (hito 2)
- `docs/CODEX_WORKFLOW.md` — cómo trabaja Claude con el motor
- `docs/AUDIO_ENGINE.md` — composición, plantillas, renderizador, congelado y calibración
- `docs/ANALYSIS.md` — análisis, separación, confianza y verificación
- `docs/PROJECT_SCHEMA.md` — modelo, unidades, operaciones y revisiones
- `docs/COMPATIBILITY.md` — resultados del doctor, tolerancias y hallazgos
- `docs/ROADMAP.md` — etapas y compuertas
- `AGENTS.md` — reglas para Claude al trabajar en este repo
