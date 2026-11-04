# Versión DJ: extended mix y cues

```bash
uv run osky-remix extend <proyecto> --from REV --branch NOMBRE [--intro-bars 32] [--outro-bars 32]
uv run osky-remix render <proyecto> --rev NOMBRE
uv run osky-remix cues   <proyecto> --rev NOMBRE --audio master.wav --out cues.xml [--format rekordbox|csv|json]
```

## Extended mix (`dj/extend.py`)

`extend` no toca la revisión base: crea la rama indicada con un único lote atómico de
operaciones del contrato.

1. **`insert_time`** en el tick 0 por la duración de la intro. Todo lo existente se desplaza en
   bloque: clips (también de pistas bloqueadas y editadas a mano), automatización, marcadores,
   tempo y compás. Nada se regenera.
2. **Intro `intro_dj`** (16, 32 o 64 compases; frases enteras):
   - primera mitad: bombo en cada tiempo + hats (intensidad 1);
   - segunda mitad: entra el bajo; la batería sube a palmas/hat abierto (intensidad 2) y hace
     un redoble en el último compás, que desemboca en la intro original.
3. **Outro `outro_dj`** al final de la última sección, en espejo: bombo + batería + bajo; en la
   segunda mitad sale el bajo y quedan bombo y hats hasta el final.
4. Marcadores `intro_dj` / `outro_dj`, `meta.arrangement` con las dos secciones nuevas (energía 1,
   sin voz) y `meta.dj_extended` (compases, ticks, tempo de intro y outro, pistas usadas).

Decisiones:

| Tema | Decisión |
|---|---|
| Armonía del bajo | Raíz más grave por medio compás de los primeros 8 compases con bajo (intro) y de los últimos 8 (outro), en bucle. Sin bajo existente: raíz de la tonalidad; sin tonalidad, sin bajo. Patrón y registro de la plantilla (`meta.compose.template`: el de la sección de más energía); si no hay, rodante en 33–45. |
| Pistas | Bombo en `kick`, batería en `drums`, bajo en `bass`. Una pista bloqueada no recibe nada (se desplaza intacta) y el contenido va a una pista nueva `dj-drums` (batería integrada) o `dj-bass` (sintetizador integrado). |
| Tempo estable | La intro hereda el tempo del tick 0 y el outro el último tramo. Se rechaza un proyecto con un cambio de tempo en o después del final del tema. No hay voz: solo se escribe en pistas de bombo, batería y bajo; si un clip de audio existente entra en el outro, se avisa. |
| Automatización generada | En las pistas de la intro (p. ej. el highpass de la batería, que en la intro original barre de 3 kHz a 180 Hz) el carril toma en la intro y el outro DJ el valor que tiene al empezar la sección de más energía, para que el ritmo suene completo. La automatización manual o de pistas bloqueadas solo se desplaza. |
| Procedencia | El contenido generado desplazado conserva su condición de generado con la huella nueva; lo manual o editado conserva su huella antigua (sigue sin ser reemplazable). Intro y outro quedan registrados como generados. |
| Compás | Solo 4/4. |
| Repetición | Una revisión extendida no se vuelve a extender ni admite `vary` (rechazado): se varía la base y se extiende otra vez. |

### Operación `insert_time`

`{"op": "insert_time", "at_tick": T, "ticks": N}`: inserta compases vacíos en un límite de compás
(N múltiplo del compás vigente). Todo lo que empieza en `T` o después se desplaza; los puntos de
tempo y de compás posteriores también. Un carril de automatización que cruza `T` conserva su
valor en `T` durante el tiempo insertado. Se rechaza si cortaría un clip.

Es la única operación que mueve pistas bloqueadas: lo hace rígidamente, como "insertar tiempo"
en un DAW, y su postcondición comprueba que cada clip quedó idéntico salvo `start_tick`.

## Cues y beatgrid (`dj/cues.py`)

Cada posición sale de `tempo_map().tick_to_sample(tick) / sample_rate`, la misma muestra entera en
la que el renderizador coloca ese tick: nunca de segundos acumulados con bpm en coma flotante.
Se escriben con 6 decimales (1 µs, menos de una muestra a 48 kHz).

**Rekordbox XML** (`DJ_PLAYLISTS`, se importa como "rekordbox xml"):

- Un `TEMPO` por punto del mapa de tempo: `Inizio` (s), `Bpm`, `Metro` (`4/4`) y `Battito`
  (tiempo del compás en que empieza el tramo, 1–4). Un cambio que no cae en un tiempo se avisa.
- `POSITION_MARK` `Type="0"`: un memory cue (`Num="-1"`) por marcador de sección o `cue`, y hasta
  8 hot cues (`Num` 0–7 = A–H) en orden de tiempo. Si hay más de 8 secciones, entran primero la
  primera intro, el primer drop, la primera sección con voz, el último outro y el primer break, y
  el resto por orden de tiempo.
- `TRACK` con `Location` (ruta absoluta del audio), `TotalTime`, `SampleRate`, `AverageBpm`
  (promedio ponderado por tiempo) y una lista de reproducción "Osky Remix".

Colores por categoría (la categoría sale del nombre y de `meta.arrangement`):

| Categoría | Secciones | Color |
|---|---|---|
| intro | `intro*` | verde (40, 226, 20) |
| voz | sección con voz (`voz`, `verso`, `pre`, `coro2`…) | azul (48, 90, 255) |
| break | `break*`, `build*` o sin voz | violeta (180, 50, 255) |
| drop | energía 3 o `drop*` | rojo (230, 40, 40) |
| outro | `outro*` | naranja (224, 100, 27) |
| cue | marcador `kind: cue` | amarillo (195, 175, 4) |

**CSV / JSON**: la misma información (tick, compás, muestra, segundos, bpm, color) para otras
herramientas. Serato no se exporta: guarda los cues como etiquetas binarias dentro del propio
archivo de audio, algo que este motor no escribe.

`export` incluye `dj/cues.xml` (apunta a `audio/master.wav` del paquete) y `dj/cues.csv`.

## Verificación

`tests/unit/test_dj.py` y `tests/audio/test_dj_render.py` (instrumentos integrados):

- intro y outro de exactamente N compases en ticks (16/32/64); la voz desplazada exactamente
  la intro; sin clips de audio en intro ni outro; bombo en cada tiempo y bajo desde la mitad;
- pista bloqueada idéntica salvo el desplazamiento (clips y automatización); clip manual y clip
  editado conservados y sin registrarse como generados; marcadores nuevos;
- render: el master crece exactamente `tick_to_sample(fin nuevo) − tick_to_sample(fin viejo)`
  muestras; los stems de voz están en silencio en intro y outro; el bajo calla en la primera
  mitad de la intro; la pista bloqueada tiene el mismo RMS por compás (< 0,1 dB);
- cues: se vuelve a leer el XML y BPM, `Inizio` y cada cue coinciden a menos de 1 muestra con
  las fracciones exactas, también con un cambio de tempo; el inicio real de la voz en el stem
  cae a ≤ 2 muestras del cue `voz`.

Sin verificar: la importación en Rekordbox real (no está en este equipo). El formato sigue la
especificación pública del XML de Rekordbox; Rekordbox exporta los tiempos con 3 decimales y
debería leer 6.
