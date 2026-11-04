# Esquema del proyecto y contrato de operaciones

Esquemas JSON generados del modelo: `schemas/project.schema.json` y
`schemas/operation.schema.json` (`uv run osky-remix schema export`; un test falla si se
desfasan).

## Unidades

| Qué | Unidad | Tipo |
|---|---|---|
| Posición musical (clips, notas, automatización, marcadores, tempo) | ticks, PPQ 960 | entero |
| Posición dentro de un audio (offset, longitud, fades) | muestras | entero |
| Conversión | `TempoMap`: tempo constante por tramos, aritmética con `Fraction` | exacta |

Los segundos nunca se guardan como posición. `tick_to_sample` redondea una sola vez al final.

## Modelo (`core/models.py`)

- **Project**: `name`, `sample_rate`, `ppq`, `key`, `tempo[]`, `meter[]`, `assets{sha256: Asset}`,
  `tracks[]`, `automation[]`, `markers[]`. Siempre existe exactamente una pista `master`.
- **Asset**: identificado por SHA-256 del contenido; `kind` (original, stem, sample, render, midi,
  preset, plugin_state, analysis); metadatos de audio; **Provenance** (herramienta, versión, modelo,
  parámetros, semilla, assets padre).
- **Track**: `audio | instrument | bus | return | master`; clips, cadena de plugins, envíos,
  sidechain, salida, ganancia, paneo, mute/solo y `locked`.
- **AudioClip** (`start_tick`, `source_offset`/`source_length` en muestras, fades) y
  **MidiClip** (`start_tick`, `length_ticks`, notas relativas al clip).
- **PluginSlot**: `backend` (surge_xt, vst3, builtin, faust), `uri`, preset/estado como assets,
  parámetros.
- **AutomationLane**: `target` = `gain_db`, `pan` o `<plugin>.<parámetro>`; puntos ordenados.

La validación de integridad rechaza: ids duplicados, referencias a pistas/assets inexistentes,
clips en pistas del tipo equivocado, clips más largos que su asset, notas fuera del clip y
ruteos circulares (salidas, envíos y sidechain).

## Operaciones (`core/operations.py`)

`add_track`, `remove_track`, `set_track`, `lock_track`, `unlock_track`, `add_send`, `remove_send`, `register_asset`,
`add_clip`, `move_clip`, `delete_clip`, `transpose_midi`, `set_midi_notes`, `add_plugin`,
`replace_plugin`, `remove_plugin`, `set_plugin_param`, `set_automation`, `delete_automation`,
`set_tempo`, `set_meter`, `set_key`, `set_meta`, `set_master_profile`, `add_marker`, `remove_marker`.

`meta` guarda información no sonora: `compose` (cómo se compuso: análisis, voz, inicio, compases,
tempo, semilla, plantilla) y `arrangement` (variante y secciones con su energía).

Cada una: **precondiciones** → aplicar sobre copia → **revalidación completa** →
**postcondiciones**. Un lote (`osky-remix op apply`) es atómico: si falla la operación *n*, se
informa su índice y el proyecto no cambia.

### Protección de ediciones manuales

- Ninguna operación puede tocar una pista `locked` (clips, plugins, automatización, borrado).
- Solo `--author user` puede desbloquear.
- `set_tempo` se rechaza si hay pistas bloqueadas con clips (moverían su posición real).
- `revert` de Claude se rechaza si alteraría una pista bloqueada.
- `merge` reaplica operaciones: si alguna toca una pista bloqueada, es conflicto y no se escribe nada.

## Revisiones (`core/revisions.py`, SQLite)

- Historial inmutable: cada revisión guarda operaciones, snapshot canónico y su hash; las filas
  nunca se modifican. Un snapshot alterado se detecta al cargar (`corrupta`).
- Ramas con cabeza móvil; referencias por nombre de rama, id o prefijo único.
- `revert` crea una revisión nueva con el contenido antiguo (no reescribe historial).
- `merge`: ya contenida → nada; ancestro → avance rápido; divergente → reaplica las operaciones
  de la rama fuente desde el ancestro común sobre la cabeza actual, atómicamente.
  Limitación conocida: una rama con `revert` no se puede reaplicar (se informa como conflicto).

## En disco

```
workspace/projects/<nombre>/
  project.db   revisiones, ramas, metadatos (WAL, BEGIN IMMEDIATE)
  assets/      <sha[:2]>/<sha>.<ext>, solo lectura
  cache/       resultados derivados por clave de contenido (tarea, versión, entradas, parámetros)
```

### Migraciones

La integridad de una revisión se verifica hasheando el snapshot tal como se guardó, no su
re-serialización: así los campos nuevos con valor por defecto (p. ej. `master_profile`) no invalidan
revisiones anteriores.
