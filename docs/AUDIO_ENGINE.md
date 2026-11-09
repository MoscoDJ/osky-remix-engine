# Motor de audio, producción y mezcla (etapas 3 y 4)

```bash
uv run osky-remix compose <proyecto> --analysis <sha> --vocals <sha-stem> --at 1:23.1 --bars 32 \
    [--template melodic_techno] [--bpm 124] [--seed 1] [--branch idea]
uv run osky-remix render <proyecto> [--rev rama|id] [--profile club|streaming|dynamic]
```

## Composición (`composition/`)

`compose` no toca el proyecto directamente: el arreglador produce un **lote de operaciones** del
contrato (etapa 1) que se aplica de forma atómica en una revisión nueva (en la rama indicada).

1. **Contexto** (`context.py`): del análisis toma el tiempo fuerte más cercano a `--at`, el tempo
   del tramo, la tonalidad local y los acordes. Mapeo **1 tiempo del original = 1 tiempo del
   remix**: la voz se estira ×(bpm destino / bpm origen) con Rubber Band (formantes conservados)
   y la armonía del original, leída por beats, queda exactamente bajo la voz.
2. **Melodía**: Basic Pitch transcribe la voz estirada (con filtro de armónicos) y el lead la toca
   cuantizada a semicorcheas, monofónica, en el registro de la plantilla.
3. **Generadores**: batería por intensidad (0-3) con redobles y crash; bajo rodante/contratiempo/
   sostenido sobre la raíz por medio compás; pads con conducción de voces; arpegios; FX
   (riser, impacto). Todo determinista por semilla.
4. **Plantilla** (`templates/<género>/template.json`): presets, mezcla (ganancia, paneo, efectos,
   sidechain) y arreglos por longitud (secciones con qué suena en cada una).

Verificación de alineación (hito 1, "Tu Última Canción"): la proporción de la voz que cae en
notas del acorde es máxima sin desfase (67 %) y cae al desfasar medio compás (58 %, 54 %) o uno
completo (44 %, 31 %).

### Regiones, cortes y ganchos (plantilla)

Campos por sección, además de los básicos:

| Campo | Efecto |
|---|---|
| `source: {"lyric": "Creí ver", "occurrence": 1}` | la sección toma armonía, voz y melodía de esa línea (MIDI de referencia); anacrusa medida en la voz |
| `source: {"at": "2:52"}` | región por tiempo (tiempo fuerte más cercano) |
| `vocal_bars` | compases de voz dentro de la sección (evita arrastrar la anacrusa de la frase siguiente) |
| `lead: "midi:brass"` | el lead toca un canal del MIDI de referencia (aquí, la trompeta) en esa región |
| `chops: [{"lyric": "Solo tu canción", "occurrence": 2, "at_bars": [6, 14]}]` | cortes de voz en la pista `chops` (eco y reverb por envío) |
| `kick_bars` | el bombo se detiene tras N compases (outros) |

`compose --reference <sha> --harmony auto|midi|audio`: con `auto`, la armonía viene del MIDI si la
confianza de la referencia es ≥ 0.7, y del análisis del audio donde el MIDI no tiene acorde.

El arreglo de 144 compases de `melodic_techno` (≈ 4:40) usa todo esto: intro, verso, pre (con la
trompeta del original respondiendo), break con el coro, drop con la melodía del coro, drop2 con el
solo de trompeta, puente con cortes, coro 2 y outro con "Nada quedó… solo tu canción".

## Renderizador (`audio/renderer.py`)

| Elemento | Comportamiento |
|---|---|
| Clips de audio | desde el asset, offset/longitud en muestras, ganancia, fades; remuestreo si hace falta |
| Instrumentos | primer plugin de la pista: Surge XT (DawDreamer) o integrado (`drums`, `fx`, `synth`) |
| Efectos integrados | highpass, lowpass, eq, compressor, reverb, delay (en tiempos), chorus, gain, sidechain, stereo, deesser, clipper |
| Enrutado | salidas a buses/master, envíos post-fader a retornos, sidechain desde otra pista |
| Master | perfil de `mastering.dynamics` (LUFS objetivo + techo de true peak con lookahead) |
| Salidas | premaster float32, master WAV 24 bits, stems post-fader float32, reporte JSON |

**Congelado**: cada render de Surge se guarda en la caché con clave de contenido (notas, preset,
parámetros, sr, longitud). La mayoría de los patches de fábrica son estocásticos; con el
congelado, re-renderizar la misma revisión da audio idéntico bit a bit, y cambiar solo la mezcla
no vuelve a sintetizar (24 s → 12 s en el hito 1).

**Calibración**: muchos presets suenan una octava abajo de la nota MIDI. La primera vez que se usa
un preset se mide su f0 con una nota de referencia y la transposición (múltiplo de 12) se
compensa en el MIDI; se guarda en caché por SHA del preset.


## Mezcla y render (etapa 4)

```bash
uv run osky-remix graph   <proyecto> [--rev X]                    # cadena, salidas, envíos, sidechain, orden
uv run osky-remix render  <proyecto> [--rev X] [--profile P]       # por defecto, el perfil del proyecto
uv run osky-remix compare <proyecto> <A> <B> [--profile P]         # A/B igualado + diff del proyecto
uv run osky-remix export  <proyecto> <carpeta> [--rev X]           # paquete de entrega
uv run osky-remix package verify <carpeta>
```

| Elemento | Detalle |
|---|---|
| Grafo DSP | `audio/graph.py`: orden topológico por salidas, envíos y sidechain. Un ruteo circular es inválido en el modelo, así que la operación que lo crearía se rechaza. |
| Envíos | `add_send` / `remove_send`; pre-fader (tras efectos, antes de ganancia/paneo) o post-fader. |
| Automatización | `gain_db`, `pan`, `<plugin>.<parámetro>` de efectos integrados (por bloques de 512 muestras, el estado del plugin se conserva) y de Surge XT (parámetro por nombre). Curvas `linear` o `hold`. Un destino no automatizable produce un aviso y se aplica el valor fijo. |
| Filtros | Escalera de 12/24 dB/oct con resonancia (`audio/ladder.py`): la topología del `LadderFilter` de JUCE sin su saturación `tanh`, que coloreaba según el nivel (COMPATIBILITY 18). Frecuencia y resonancia automatizables por muestra. |
| Clipper | `clipper`: lineal hasta `knee·techo`, `tanh` encima, sobremuestreo 4x contra aliasing. El agente lo usa en el bus de batería (`auto-drum-clip`) antes de ceder volumen. |
| Master | Bloqueo de DC a 5 Hz tras el limitador, dentro del bucle de true peak (COMPATIBILITY 19). |
| Estéreo | efecto `stereo`: ancho por M/S y graves en mono. |
| Gain staging | aviso si una pista supera 0 dBFS antes del fader; los instrumentos integrados dejan 6 dB de margen. |
| Mastering | perfiles en `presets/mastering/*.json` o integrados: bloqueo de DC, pico de entrada, graves en mono, tilt, glue, objetivo LUFS y techo de true peak. Campo `master_profile` del proyecto (`set_master_profile`). |
| Calidad | `mastering/quality.py`: true peak, LUFS, recortes, NaN, DC, compatibilidad mono, silencio inicial, LRA; estado PASS/WARN/FAIL. |
| A/B | `mastering/compare.py`: B igualado a la sonoridad de A; bandas, cresta, estéreo, null test, por sección (marcadores) y `core/diff.py`. |
| Entrega | master 24 bits, premaster float, stems alineados, MIDI (PPQ 960, mapa de tempo), presets, `plugins.json`, calidad, proyecto + assets y `manifest.json` con SHA-256. |

Esquemas: `schemas/render.schema.json` describe el reporte de render.

Sesión editable para un DAW: `osky-remix dawproject` (también `daw/` en el paquete), ver `docs/DAWPROJECT.md`.


## Semántica de audio y entregas (2026-11-04)

`source_offset`, `source_length` y fades se expresan en muestras del asset. Se lee solo el
fragmento fuente, se remuestrea y su longitud de destino se redondea hacia arriba, como
`resample_poly`. API y render usan la misma conversión; el inicio se obtiene del mapa de tempo.

Solo, como en un DAW:

- **Pista o bus en solo:** suenan ella, lo que le llega por su salida (las fuentes de un bus) y
  todo lo que hay río abajo (sus salidas y envíos, p. ej. su reverb).
- **Retorno en solo:** las pistas que le envían señal se procesan para alimentarlo, pero su
  salida directa calla; en el master solo está el retorno. Lo mismo si la señal llega por una
  cadena (pista → bus → envío al retorno).
- Una pista fuera del solo no alimenta un retorno que solo es audible por estar río abajo
  (el envío del bombo al reverb de la voz en solo calla), ni una alimentadora llega por un
  segundo envío a otra pista audible.

Varias pistas solo se suman. Mute prevalece y silencia también los envíos pre-fader;
el detector de sidechain conserva la señal dry (una pista fuera del solo sigue disparándolo). Bypass de instrumento MIDI deja de generar
señal; bypass de un efecto permite pasar la señal sin ese efecto.

El mastering distingue `converged`, `silent`, `unmeasurable` y `not_converged`. Los niveles
no medibles se serializan como null. Un preview puede contener un resultado fallido con su
reporte; `export` rechaza mastering no convergente, calidad FAIL y capacidades de plugin no
implementadas. Un puntaje del agente es una medida técnica, no una aprobación musical.

Benchmark reproducible: `uv run python scripts/benchmark_render.py --seconds 180 --tracks 8`.
En este equipo: 12.70 s y 2596.3 MiB RSS máximo, -14.00 LUFS, -11.46 dBTP. Fixture sintético
con ocho pistas de audio, sin instrumentos externos: no predice el coste de plugins pesados.
Se mantiene el procesamiento en memoria; dimensionar RAM antes de sesiones mayores.


## Voz: de-esser y transposición

### De-esser (`audio/deesser.py`, efecto integrado `deesser`)

Banda dividida, sin latencia:

| Parámetro | Por defecto | Detalle |
|---|---|---|
| `threshold_db`* | −30 | umbral sobre el RMS de la banda (dBFS, estéreo enlazado) |
| `range_db`* | 6 | reducción máxima de la banda |
| `freq_hz`, `width_oct` | 6700, 0.85 | centro y ancho de la banda (≈ 5–9 kHz) |
| `ratio` | 8 | pendiente por encima del umbral (valores altos ≈ limitador) |
| `attack_ms`, `release_ms` | 0.5, 60 | suavizado de la reducción en dB |

La banda se aísla con un pasabanda Butterworth de fase cero (`sosfiltfilt`); el resto es
`x − banda`, su complemento exacto, y solo la banda se multiplica por la ganancia. Así el audio
no se retrasa (`LATENCY_SAMPLES = 0`) y fuera de la banda nada cambia. El detector usa
ventanas centradas de 1 ms cada 32 muestras (un pequeño lookahead del detector, nunca del
audio) y la ganancia se interpola a resolución de muestra. Sin sibilancia por encima del
umbral la salida es idéntica bit a bit a la entrada. `threshold_db` y `range_db` aceptan
automatización (`<plugin>.threshold_db`) por muestra.

Pruebas (`tests/audio/test_deesser.py`): tono armónico 220–1760 Hz + ráfagas de ruido de
6.5–7.5 kHz en tiempos conocidos. Las ráfagas bajan `range_db` ± 0.3 dB (3, 6 y 10 dB), la
banda 200–2000 Hz cambia < 0.2 dB sin desfase, los bordes de las ráfagas no se mueven, la
señal sin sibilancia queda intacta y el resultado es determinista.

### Transposición (`audio/pitch.py`, `osky-remix transpose`)

```bash
uv run osky-remix transpose <proyecto> <sha> --semitones -2 [--no-formant] [--clip v1]
uv run osky-remix transpose <proyecto> <sha> --to-key proyecto --analysis <sha-análisis> --clip v1
uv run osky-remix transpose <proyecto> <sha> --to-key "C minor" --from-key "A minor"
```

- Rubber Band R3 (`--fine`), duración intacta (si difiere, se ajusta a la longitud exacta del
  original para que los clips sigan siendo válidos) y formantes conservados (`-F`) por defecto.
- No destructivo: el resultado es un asset **nuevo** con procedencia (`rubberband`, versión,
  `semitones`, `formant`, `engine`, padre = asset original). En caché por (sha, semitonos,
  opciones, versión de Rubber Band); repetir la misma transposición no vuelve a procesar.
- `--to-key`: tonalidad destino (`proyecto` = `Project.key`, la que fija `compose` con
  `set_key`). La de origen viene de `--from-key` o de `harmony.key` de un reporte de análisis.
  `transposition_to_key` elige el desplazamiento mínimo en −6…+5: con modos iguales se alinean
  las tónicas (A minor → C minor = +3); con modos distintos se compara la relativa
  (A minor → C major = 0, mismas notas). El tritono se resuelve siempre hacia abajo
  (F# minor → C minor = −6).
- `--clip`: la misma revisión registra el asset y aplica `set_clip_asset`, que cambia solo el
  asset del clip (posición, recorte, ganancia y fades se conservan) y exige la misma frecuencia
  de muestreo y longitud suficiente. El asset original sigue registrado.

Pruebas (`tests/audio/test_pitch.py`, `tests/unit/test_set_clip_asset.py`): 440 Hz + 2
semitonos mide 493.88 Hz ± 0.5 % (`f0_median`), misma duración, original intacto (SHA),
acierto de caché en la segunda llamada, distancias entre las 24 × 24 tonalidades.

Corrección de afinación a escala: no implementada. Requiere desplazamiento variable en el tiempo
(mapa de tono de Rubber Band) guiado por f0 por nota; queda pendiente hasta poder verificarla
con fixtures de afinación conocida.
