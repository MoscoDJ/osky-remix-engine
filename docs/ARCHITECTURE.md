# Arquitectura

Tres responsabilidades separadas:

1. **Claude decide qué producir** (`agents/`): planifica y emite operaciones.
2. **El motor musical representa la producción** (`core/`, `composition/`, `analysis/`):
   proyecto canónico, timeline en ticks (PPQ 960) y muestras enteras, operaciones atómicas
   validadas con precondiciones y postcondiciones.
3. **El motor de audio ejecuta cómo suena** (`audio/`, `mastering/`, `separation/`):
   backends intercambiables (Surge XT vía DawDreamer, síntesis propia, Faust/Csound;
   Demucs/RoFormer; perfiles de mastering).

Claude no edita el archivo del proyecto directamente: toda modificación pasa por el contrato
de operaciones (`schemas/operation.schema.json`, etapa 1).

Los plugins y modelos se instalan por separado y se validan con `osky-remix doctor`. Si un
plugin falla, se registra, se aísla (cada prueba corre en su propio proceso) y se usa un
backend alternativo cuando exista.

- **Biblioteca de samples** (`samples/`): `AssetStore` propio en `workspace/samples/assets`,
  rasgos (`samples.features`, sin modelos) en `CacheStore` por sha + `FEATURES_VERSION`, índice
  SQLite `library.db`. Un kit (`samples.kit`) es un asset JSON nota → sample referenciado por
  `PluginSlot.preset` de los instrumentos integrados `drums` y `sampler`; el renderer exige que
  sus samples estén registrados en el proyecto (`validate_capabilities`).
  Opciones por voz: `pitch_st` (remuestreo) y `anchor` (`end`: el sample termina en el fin de
  la nota); `fx` también acepta kit. `samples.sounds` resuelve un juego de sonidos (presets de
  usuario + kits) a assets del proyecto y genera las operaciones de cambio con igualación de
  sonoridad por pista (`sounds apply`).
- **Lotes** (`batch/`): pasos enchufables `fn(ctx) -> dict` que llaman en proceso a las
  funciones de la CLI (`--json`); manifiesto JSON con bloqueo `fcntl` y escritura atómica por
  canción/paso; reanudación saltando pasos hechos; `--jobs` con procesos `spawn`, separación
  serializada por un bloqueo global y renders con los bloqueos de la caché.

## Correcciones de fiabilidad (2026-11-04)

- La reversión HTTP requiere `branch` y `expected_head`; las operaciones manuales pueden
  enviar `expected_head`. Las confirmaciones del planificador usan la revisión evaluada.
- Cada trabajo fija revisiones completas y el contenido del perfil al enviarse. Se ejecuta
  en un proceso `spawn` separado: una caída nativa, cancelación o timeout no termina el servidor.
  Solo un servidor puede poseer una cola; los trabajos interrumpidos se marcan fallidos y los
  pendientes se retoman. Timeout predeterminado: 1800 segundos.
- CLI A/B y API comparten `audio.render_service`: la identidad incluye revisión completa,
  perfil resuelto, motor, versiones DSP y hash del binario Surge. El render explícito de CLI
  permite elegir carpeta y stems, y no reutiliza una salida previa.
- Las entradas de caché son inmutables. Publicación con temporal único, bloqueo de proceso
  por clave y renombrado; los renders verifican además hashes de los archivos publicados.
- `meta.generated_content` guarda huellas del contenido generado. Regenerar conserva
  pistas con clips o lanes manuales/editados y marcadores manuales. Proyectos antiguos sin
  procedencia se conservan de forma prudente: no se deduce propiedad por nombres.
- La separación incorpora versiones externas y hashes de los modelos locales a su identidad;
  los modelos disponibles verificados se fijan en `requirements/model-hashes.json`. Un modelo
  nuevo debe revisarse y añadirse al manifiesto antes de considerarlo una referencia fijada.

La cola aísla el audio de la API. En CLI, cada invocación ya es un proceso separado del servidor.
No se ha convertido el motor en un procesador de audio en tiempo real ni por bloques.
