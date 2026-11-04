# Compatibilidad (etapa 0)

Fuente de verdad: `osky-remix doctor`. Cada prueba verifica el audio que produce; terminar
con código 0 no basta. Reporte JSON en `workspace/reports/`.

## Equipo auditado (2026-11-03)

| Componente | Versión / detalle |
|---|---|
| SO | Ubuntu 26.04.1 LTS (Kubuntu), x86_64, 32 hilos, 78 GB RAM |
| GPU | NVIDIA GeForce RTX 2070, 8 GB, driver 595.99.02, cc 7.5 |
| Python del proyecto | 3.11.17 (uv). 3.12 descartado: `basic-pitch` no resuelve |
| Surge XT | 1.3.4, `.deb` oficial de GitHub (sha256 `a6e55064…bb151`), VST3 en `/usr/lib/vst3` |
| DawDreamer | 0.9.0 |
| Demucs | herramienta independiente (`uv tool install demucs`), torch con CUDA |
| Rubber Band | 4.0.0 (apt `rubberband-cli`) |
| FFmpeg | 8.0.1 |
| Faust / Csound | apt; Faust se usa embebido vía DawDreamer |

## Criterios y tolerancias

| Prueba | Criterio de aceptación |
|---|---|
| Python | 3.11, 15 paquetes importables, `uv.lock` presente |
| GPU | CUDA visible; conv1d en GPU = CPU con dif. < 1e-3 |
| Demucs | 4 stems de longitud exacta; reconstrucción ≥ 15 dB; correlación stem↔fuente conocida ≥ 0.7 |
| Surge XT | preset de fábrica cargado; pico > -40 dBFS; f0 a ±30 cents; espectro distinto al patch por defecto |
| DawDreamer | automatización cambia el nivel > 6 dB; estado guardado → audio idéntico al recargar; efecto VST3 procesa audio |
| Rubber Band | tempo ×1.5: duración ±1 %, tono ±15 cents; +3 semitonos ±15 cents |
| Basic Pitch | ≥ 7/8 notas (tono exacto, inicio ±70 ms), ≤ 1 nota extra tras filtrar armónicos |
| FFmpeg | WAV24→FLAC→WAV bit-exacto; metadatos legibles; remuestreo con duración exacta |
| Mastering | EBU 3341 casos 1-2 a ±0.1 LU; true peak de fs/4 a ±0.2 dB; perfil `club` ≤ -1 dBTP, ±0.5 LU, 0 recortes |
| Reproducibilidad | patch determinista: bit-exacto entre procesos (tol. -120 dBFS). Patch estocástico: ≤ 0.1 LU y ≤ 1 dB en bandas de 1/3 de octava |

## Hallazgos (afectan el diseño)

1. **Presets .fxp en Surge VST3.** `load_preset` devuelve False (formato VST2). El estado VST3
   de Surge es el bloque `sub3` del .fxp: `plugin_host.fxp_to_state_file` lo envuelve en el
   formato de estado de JUCE (`VC2!` + XML + base64 propio de JUCE) y se carga con `load_state`.
2. **Carga asíncrona del patch.** Tras `load_state`, Surge descarta las notas de los primeros
   instantes y pisa cualquier `set_parameter`. Política: pre-roll de 0.5 s (`PREROLL_S`) y
   parámetros estáticos como automatización constante (`set_static_params`).
3. **Verificación de estados.** `load_state` devuelve None y `get_parameter` muestra la caché
   del host, no lo cargado. Los estados se verifican siempre por audio.
4. **Determinismo de Surge.** Algunos patches (el de por defecto, `Keys/EP 1`) tienen
   aleatoriedad interna: dos renders difieren muestra a muestra aunque suenan igual
   (0.06 LU, 0.33 dB). Otros (`Leads/Acidofil`, `Basses/Attacked`) son bit-exactos.
   Política: las pistas con patch estocástico se congelan como asset (SHA-256).
5. **Limitador.** `pedalboard.Limiter` (JUCE) compensa ganancia y recorta en 0 dBFS, sin
   lookahead ni true peak (+1.4 dBTP, 10 337 muestras recortadas en la prueba). Se reemplazó
   por `mastering.dynamics.limit`: detección true peak 4× con lookahead y garantía
   ganancia[n] ≤ techo/pico[n].
6. **Basic Pitch.** Con parámetros por defecto añade armónicos fantasma (2-3 octavas arriba,
   ~½ amplitud): 5 notas extra en 8. `analysis.transcribe` los filtra (0 extra).
   Requiere `setuptools<81` (usa `pkg_resources`).
7. **Csound** carga con un aviso inocuo (`liblinear_algebra.so: undefined symbol dorgqr_`).
8. **TensorFlow** no se instala: Basic Pitch corre con el modelo ONNX.
