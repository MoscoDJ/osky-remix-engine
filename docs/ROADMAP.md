# Roadmap

Regla: no se avanza de etapa si los criterios críticos de la anterior no se cumplen.
Cada etapa entrega una demostración funcional, pruebas automatizadas y documentación.

| Etapa | Contenido | Estado |
|---|---|---|
| 0 | Auditoría del entorno: `osky-remix doctor` | **Completa** (13/13 PASS, 2026-11-03) |
| 1 | Núcleo y CLI: proyectos, assets SHA-256, esquemas Pydantic, revisiones, operaciones atómicas, caché | Siguiente |
| 2 | Análisis y separación: BPM, compás, tonalidad, estructura, stems, transcripción, reportes con confianza | |
| 3 | Producción musical: MIDI, síntesis (Surge XT), arreglos, voces, plantillas de género | |
| 4 | Mezcla y render: grafo DSP, buses, sidechain, mastering por perfiles, exportación | |
| 5 | Automatización con Claude: planificador, acciones, evaluador, variaciones, revisiones | |
| 6 | Interfaz híbrida: timeline, mezclador, editor MIDI, A/B, control manual | |

## Hitos musicales

1. 32 compases de melodic techno con voz extraída de una canción de prueba, bajo y batería
   sintetizados, mezcla básica y render WAV.
2. Tres arreglos distintos del mismo material sin perder ninguna edición manual.

## Ya disponible desde la etapa 0

- `audio.plugin_host`: carga de presets .fxp de Surge, pre-roll, parámetros estáticos.
- `mastering`: LUFS (BS.1770), true peak 4×, limitador con lookahead, perfiles `club`,
  `streaming`, `dynamic`.
- `analysis.transcribe`: Basic Pitch ONNX + filtro de armónicos.
