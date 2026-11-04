# Análisis y separación (etapa 2)

```bash
uv run osky-remix ingest   <proyecto> cancion.flac           # original intacto + copia PCM float32
uv run osky-remix separate <proyecto> <sha> -b demucs --lead-backing
uv run osky-remix analyze  <proyecto> <sha> --stems <sha-reporte-separación> --md analisis.md
```

Todo resultado se registra como asset con procedencia (herramienta, versión, modelo, parámetros,
padres) y se guarda en caché por clave de contenido. Cambiar el algoritmo implica subir
`ANALYSIS_VERSION`, lo que invalida la caché sin borrar nada.

## Estimaciones y confianza

Cada estimación es `Estimate{value, confidence 0-1, alternatives, method, notes}`.

| Qué | Método | Confianza |
|---|---|---|
| Tempo | envolvente de ataques multibanda (peso en graves) → zonas de tempo local → beat tracking por zona → ajuste lineal por tramos (conteo secuencial de beats) → corrimiento fino con la derivada de energía de la onda | residuo de la rejilla, acuerdo con Essentia, confianza de Essentia |
| Tempo libre | tramos cuyo residuo supera el 4 % del periodo (rubato) o con pulso débil en el tempograma; no reciben rejilla | — |
| Compás | autocorrelación de acentos por beat (graves, cambio armónico, ataque) a 3 vs 4 tiempos | diferencia entre periodicidades |
| Tiempo fuerte | fase de mayor acento por ventanas de 8 compases, por tramo de tempo; un cambio sostenido 2 ventanas es un compás irregular | margen mediano entre la mejor y la segunda fase |
| Tonalidad global | voto de 4 perfiles de Essentia KeyExtractor | fracción de votos × fuerza |
| Tonalidad por tramos | Krumhansl por ventanas de 20 s con histéresis → reetiquetado por acordes contenidos → frontera movida al cambio de acorde que mejor separa ambas | ajuste diatónico de los acordes |
| Acordes | croma CQT (armónico HPSS) sincronizado a beats contra 24 plantillas + Viterbi con penalización por cambio | margen sobre la segunda plantilla |
| Secciones | mediana por compás de vectores por beat (croma, MFCC, RMS, graves) → novedad de tablero de ajedrez → prior de frase de 4 compases → etiquetas por similitud | altura del pico de novedad |
| Silencios, transitorios, espectro | RMS < -55 dBFS ≥ 0.3 s; ataques refinados a ±1.5 ms; LUFS, LRA aprox., true peak, bandas, estéreo | — |

## Separación

| Backend | Modelo por defecto | Stems | Nota |
|---|---|---|---|
| `demucs` | `htdemucs_ft` (float32) | vocals, drums, bass, other | `htdemucs_6s` agrega guitar, piano |
| `roformer` | BS-Roformer Viperx 1297 | vocals, instrumental | audio-separator normaliza a 0.9 y escribe 16 bits |
| `--lead-backing` | Mel-Roformer Karaoke | lead_vocals, backing_vocals | se aplica al stem de voz |

`separation.quality.prepare_stems` **mide** (no supone) desfase por correlación cruzada y
ganancia por mínimos cuadrados contra la mezcla, iguala longitudes y reporta regiones a reparar:

- `reconstruction_error`: ventanas activas donde mezcla − Σstems tiene SNR < 12 dB.
- `shared_with_drums`: la voz contiene una copia escalada de la batería (proyección por mínimos
  cuadrados > -20 dB). Sibilantes y consonantes no proyectan. Con referencia de batería la
  dirección de la fuga no es observable, de ahí el nombre.
- `drum_bleed` (sin stem de batería, HPSS > 35 %), `bass_bleed` (< 90 Hz > 25 %),
  `highfreq_bleed` en el bajo (> 1.5 kHz > 20 %).

## Verificación

Fixtures sintéticos con verdad conocida (`doctor/songs.py`): pop 4/4, vals 3/4, cambio de tempo
100→128, compás de 2 tiempos insertado, intro rubato, modulación Dm→D, estructura
intro/A/B/A/B/outro. Tests en `tests/audio/` (marcador `audio`).

### Canción real de referencia (no incluida en el repo)

"Tu Última Canción" (Los Temerarios), contra análisis manual, letra (Genius) y acordes (Cifra Club):

| Qué | Referencia | Detectado |
|---|---|---|
| Tempo | 86.095 BPM | 86.094 BPM, residuo 13.5 ms |
| Intro rubato | 0:00–0:41 sin pulso | tempo libre 0:00.5–0:37.7 |
| Tonalidad | Dm, modula a D en el coro (~2:03) | Dm hasta 2:00.8, D después |
| Acordes de estrofa | Dm Gm C F Dm Gm A Dm | Dm Gm C F Dm Gm A Dm |
| Coro 1 / interludio / coro 2 | 2:06 / 2:55–3:16 / 3:16 | 2:07.8 / 2:51–3:16 / 3:16, coros con la misma etiqueta |
| Coro 2 desfasado ½ compás | +2 tiempos | neto +2 (confianza 0.41: armonía cada 2 tiempos) |
| Separación | — | Demucs 21 dB de reconstrucción; lead/backing 71 dB; sin falsos positivos por sibilantes |

## Limitaciones conocidas

- La fase del tiempo fuerte es ambigua cuando la armonía cambia cada 2 tiempos; se reporta con
  confianza baja en vez de forzar una respuesta.
- La LRA es una aproximación (percentiles de sonoridad a 3 s), no el algoritmo EBU completo.
- audio-separator no permite salida float: sus stems llegan en 16 bits.
- Primera ejecución de RoFormer descarga el modelo (~600 MB) en `~/.cache/audio-separator-models`.

## MIDI de referencia (`analysis/midi_ref.py`)

```bash
uv run osky-remix reference <proyecto> cancion.mid --analysis <sha> --stems <sha-separación>
```

Un MIDI de terceros (p. ej. karaoke) aporta armonía exacta, contramelodías (metales, cuerdas),
bajo y letra sílaba por sílaba, pero no coincide con la grabación (otra introducción, tempo
distinto, a veces compases extra). Se alinea en dos pasos:

1. **DTW** sobre croma armónico (MIDI sin batería ni voz vs. stems bajo + otros) más actividad de
   batería (golpes del MIDI vs. ataques del stem); la batería marca entradas de sección que la
   armonía sola no distingue (intro y primer compás con el mismo acorde).
2. **Anclas por beat**: cada negra del MIDI con batería se ancla al beat del audio más cercano a
   donde la sitúa el DTW (< 0.35 beat); se completa 1 negra = 1 beat hacia atrás desde la primera
   ancla y en huecos que cuadran exactamente.

Verificación: golpes del MIDI vs. ataques del stem de batería, letra sobre actividad vocal y un
**juez independiente**: qué fracción de la voz original (la melodía está en la grabación, la
armonía en el MIDI) cae en notas de la armonía alineada, desplazando ±1/±2 tiempos. Si el máximo
no está en 0 (o −1, por anticipación melódica), la confianza se reduce a la mitad.

Ni la armonía (cambios cada 2 tiempos) ni la batería (bombo en 1 y 3) distinguen un corrimiento
de 2 tiempos; por eso hace falta el juez. Resultados en "Tu Última Canción" (voz en la armonía):

| Método | Desfase 0 | Máximo en |
|---|---|---|
| Solo DTW | 68.2 % | 0/−1 |
| Ancla al beat más cercano | 68.4 % | 0 |
| Voto por mayoría del desfase (descartado) | empeoró los coros (máximo en −2) | −2 |
| Viterbi sobre desfases con costo armónico (descartado) | 64.3 % | −1/−2 |
| **DTW con batería + anclas + completar (actual)** | **68.9 %** | **0** |

El karaoke marca cada línea en su tiempo fuerte, no en el ataque: la anacrusa se mide en la
energía del stem de voz ("Y será": 2.6 tiempos).
