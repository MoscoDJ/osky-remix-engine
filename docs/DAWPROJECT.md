# Exportación a DAWproject

```bash
uv run osky-remix dawproject <proyecto> <salida.dawproject> [--rev X] [--bounce/--no-bounce] [--json]
```

[DAWproject 1.0](https://github.com/bitwig/dawproject) es el formato abierto de intercambio que
leen Bitwig Studio, Studio One y Cubase. Sirve para abrir en un DAW un remix producido por el
motor y terminarlo a mano. Código en `audio/dawproject.py`. `export` también deja el archivo en
`daw/<proyecto>.dawproject` (con bounce) dentro del paquete de entrega.

## Contenido del archivo

| Ruta en el ZIP | Contenido |
|---|---|
| `project.xml` | estructura, arreglo y transporte (válido contra el `Project.xsd` oficial) |
| `metadata.xml` | título y comentario (válido contra `MetaData.xsd`) |
| `audio/<nombre>-<sha12>.<ext>` | assets de los clips, byte a byte (mismo SHA-256 que en el almacén) |
| `audio/bounce/<pista>.wav` | render pre-fader de cada pista de instrumento (float, frecuencia del proyecto) |
| `plugins/<pista>-<slot>.vstpreset` | estado VST3 de Surge XT construido a partir del `.fxp` |
| `presets/<pista>-<slot>.fxp` | preset original de Surge XT |
| `osky/manifest.json` | lo que no tiene equivalente, por pista, y el SHA-256 de cada archivo |
| `osky/project.json` | proyecto canónico de la revisión |
| `LEEME.md` | resumen legible para quien abre la sesión |

## Qué se traduce

| Motor | DAWproject |
|---|---|
| tempo (constante por tramos) | `Transport/Tempo` (primer punto) + `TempoAutomation` con puntos `hold` |
| compás | `Transport/TimeSignature` + `TimeSignatureAutomation` (compás → negras con el mapa de compás) |
| posiciones en ticks | unidad `beats` (negras): `beats = tick / 960` |
| pista audio / instrumento | `Track` `audio` / `notes`, canal `regular` |
| bus | `Track` `tracks` con canal `submix`; las pistas que salen al bus van anidadas dentro |
| retorno / master | canal `effect` / `master` |
| salida | `Channel@destination` (el canal de destino) |
| envíos | `Send` con `destination`, `type` `pre`/`post` y `Volume` lineal (`10^(dB/20)`) |
| ganancia / paneo / mute / solo | `Volume` lineal (máx. 2 = +6 dB o más si hace falta), `Pan` normalizado `(pan+1)/2`, `Mute`, `solo` |
| clip MIDI | `Clip` en negras con `Notes`; `vel = velocity / 127`, canal 0 |
| clip de audio | `Clip` + `Warps` (contenido en segundos) + `Audio`/`File` (ver abajo) |
| fades de clip | `fadeTimeUnit="seconds"`, muestras del asset / su frecuencia |
| marcadores | `Markers/Marker` (el tipo y el id van en `comment`) |
| automatización `gain_db` | `Points` sobre el `Volume` del canal, unidad lineal |
| automatización `pan` | `Points` sobre el `Pan` del canal, normalizado; `linear`/`hold` se conservan |
| Surge XT | `Vst3Plugin` (class ID `ABCDEF01-9182-FAEB-566D-624153675854`) con `State` `.vstpreset` |

### Clips de audio

El motor no estira audio: un clip suena a velocidad natural `source_length` muestras del asset
**a la frecuencia del asset** (el render remuestrea). Por eso el contenido se expresa en segundos
del archivo (`source_offset / sr_asset`, `source_length / sr_asset`), no del proyecto, y se
coloca en el arreglo con un `Warps` cuyos puntos van en `(0, 0)`, en cada cambio de tempo que cae
dentro del archivo y al final. Como el tempo es constante por tramos, la correspondencia
segundos → negras es lineal por tramos y resulta exacta. Con el archivo empezando en el segundo
`F = inicio − offset` del arreglo:

- `time = start_tick / 960`, `playStart = beats(inicio) − beats(F)`,
- `duration = beats(inicio + longitud) − beats(inicio)`, `playStop = playStart + duration`.

Si el archivo empezaría antes del segundo 0, se extrapola con el primer tempo. Ejemplo verificado
en las pruebas: asset a 44.1 kHz en un proyecto a 48 kHz, 120 BPM, clip en la negra 4 con offset
22050 muestras y longitud 44100 → `playStart = 1.0`, `duration = 2.0`, warps `(0,0)` y `(6,3 s)`.
Un clip que cruza un cambio a 90 BPM recibe un punto de warp intermedio.

### Instrumentos y bounce

- **Surge XT**: se escribe como `Vst3Plugin` con el estado en `.vstpreset` (formato PresetFile
  del SDK VST3, chunk `Comp` = el bloque `sub3` del `.fxp`, el mismo que usa el motor). La prueba
  `compat` lo carga con `load_vst3_preset` en un host VST3 real (DawDreamer/JUCE) y suena
  idéntico al render del motor. Los parámetros fijos (`params`) y su automatización no están en
  el estado: se listan en el manifiesto.
- **Instrumentos integrados** (`drums`, `fx`, `synth`): no existen fuera del motor; se exportan las
  notas y el instrumento queda indicado en el comentario de la pista y en el manifiesto.
- **`--bounce` (por defecto)**: cada pista de instrumento con notas se renderiza **pre-fader**
  (instrumento + efectos del motor, sin mute ni solo) y se añade al lado una pista de audio
  `"<nombre> (bounce)"` con el mismo destino, volumen, paneo, envíos y automatización; la pista
  MIDI original queda en mute. Pre-fader porque el stem post-fader (`render --stems`) llevaría la
  ganancia y el paneo, que el DAW volvería a aplicar. Si otra pista sale directamente a una pista
  de instrumento (ruteo poco habitual), su señal queda incluida en el bounce.

## Qué no se traduce (y dónde queda)

Nada se descarta en silencio: todo lo siguiente aparece por pista en `osky/manifest.json`, en el
`comment` de la pista o del clip y en `LEEME.md`.

- **Efectos integrados** (`audio/effects.py`: filtros, eq, compresor, reverb, delay, chorus, gain,
  sidechain, stereo): no tienen equivalente fiable en los dispositivos genéricos de DAWproject. Se
  listan con sus parámetros (`effects_not_mapped`). Con `--bounce` los de las pistas de
  instrumento quedan impresos en el audio; los de pistas de audio, buses y retornos no.
- **Sidechain** (`sidechain_from_not_mapped`).
- **Ganancia de clip** (`clip_gain_db_not_mapped`): DAWproject no tiene ganancia por clip.
- **Automatización de parámetros de plugin** (`automation_not_mapped`).
- **Mastering** (`master_profile_not_mapped`): el master del DAW queda sin procesar.
- Bypass de efectos: se conserva en el manifiesto.

## Diferencias conocidas

- Ley de paneo: el motor usa potencia constante con +3 dB en los extremos (`pan_gain`); cada DAW
  aplica la suya.
- Rampas de ganancia: el motor interpola en dB y el DAW en la unidad del parámetro (lineal); las
  rampas se subdividen en pasos de ≤ 1 dB (máx. 64) para que la curva coincida.
- La automatización del motor interpola por muestras; con un cambio de tempo dentro de una rampa
  el DAW (que interpola en negras) difiere ligeramente.
- Las curvas de fade del DAW pueden no ser lineales como las del motor.

## Verificación

`tests/audio/test_dawproject.py` lee el XML de vuelta y comprueba valores exactos (tempo, compás,
posiciones en negras, notas, offsets con remuestreo y cruce de tempo, ruteo, envíos, marcadores,
automatización, SHA-256 de cada archivo del ZIP) y valida `project.xml` y `metadata.xml` con
`xmllint` contra los XSD oficiales (copiados en `tests/fixtures/dawproject/`, MIT). No se ha
abierto el archivo en un DAW real: la carga en Bitwig, Studio One o Cubase queda por verificar.
