# Interfaz híbrida (etapa 6)

```bash
cd frontend && npm ci && npm run build      # una vez (o tras cambiar el frontend)
uv run osky-remix serve                     # http://127.0.0.1:8765 (solo local)
```

Desarrollo del frontend con recarga en caliente: `uv run osky-remix serve` en una terminal y
`cd frontend && npm run dev` en otra (Vite redirige `/api`, `/files` y `/ws` al servidor).

## Qué hace

| Panel | Acción | Operación del contrato |
|---|---|---|
| Timeline | arrastrar un clip (ajuste al tiempo; Shift = compás) | `move_clip` |
| Timeline / Mezclador | candado de pista | `lock_track` / `unlock_track` |
| Mezclador | fader, paneo (una revisión al soltar, o 600 ms tras la última tecla), M, S | `set_track` |
| Editor MIDI | clic en la rejilla agrega/quita notas, transponer; «Guardar» = una revisión | `set_midi_notes` |
| Reproductor | «Renderizar» (perfil del proyecto u otro) | trabajo `render` |
| A/B | dos ramas o revisiones, igualadas en sonoridad, conmutación sincronizada, bandas, secciones y diff | trabajo `compare` |
| Reproductor · visualización | espectrograma, LUFS-S con secciones y objetivo del perfil, correlación estéreo | trabajo `visualize` |
| Escucha ciega | X/Y sorteadas e igualadas, preferencia, nota 1–5 por versión, comentario; luego se revela | trabajo `blind` + `POST /listening` |
| Evaluación | energía por sección y hallazgos del agente | trabajo `evaluate` |
| Historial | revisiones de la rama (tú / claude); «volver aquí» crea una revisión nueva | `revert` |
| Cabecera | proyecto, rama, «+ rama» | `create_branch` |

En la interfaz el autor de las operaciones es **el usuario**: puede bloquear y desbloquear pistas.
Las pistas bloqueadas se muestran punteadas y no se pueden arrastrar ni editar. Un lote que el
contrato rechaza se informa con el índice y el motivo, y no se aplica nada.

## Arquitectura

- `api/app.py` (FastAPI): mismo contrato que la CLI; `/ws` difunde revisiones y estado de trabajos
  a todas las pestañas abiertas (lo que haga Claude por la CLI no aparece por WebSocket porque no
  pasa por el servidor; la interfaz recarga al cambiar de rama o tras sus propias acciones).
- `storage/jobs.py`: cola de trabajos en SQLite (`workspace/jobs.db`) con un worker local; los
  trabajos interrumpidos por un reinicio quedan como fallidos.
- `/files/…` sirve solo `.wav/.json/.md/.mid` dentro del workspace (sin salir de él).
- Frontend: React 18 + TypeScript + Vite, WaveSurfer.js 7 (forma de onda del master con regiones
  por sección); el timeline y el piano roll son SVG.

## Visualización del master

Tras cada render, la interfaz lanza el trabajo `visualize` sobre el mismo render (misma revisión y
perfil; reutiliza el render en caché). El resultado se guarda en la caché del proyecto con la clave
`cache_key("visualize", VIS_VERSION, [clave del render])` y se dibuja en canvas
(`analysis/visualize.py`, `components/Visuals.tsx`):

- **Espectrograma**: STFT Hann de 2048 muestras sobre la suma mono, ≤ 600 columnas (promedio de
  potencia) × 128 bandas logarítmicas (20 Hz–Nyquist). Se envía como matriz `uint8` en base64:
  0 = 90 dB bajo el máximo, 255 = máximo. Fila 0 = graves.
- **LUFS-S** (BS.1770-4 / EBU Tech 3341): ponderación K, ventana rectangular de 3 s, paso 0,1 s
  (más largo si la pieza superaría 3000 puntos), sin compuerta. Cada punto lleva el tiempo del
  **final** de su ventana, como un medidor en vivo: un cambio en `t` se ve entre `t` y `t + 3 s`.
  Se superponen los marcadores de sección y la línea del objetivo integrado del perfil.
- **Correlación estéreo**: Σ(L·R)/√(ΣL²·ΣR²) en ventanas de 0,4 s y por sección (+1 mono,
  −1 canal invertido, 0 sin relación; vacío en silencio).

El cursor de reproducción se dibuja sobre los tres paneles; al pasar el mouse por las curvas se
muestra el valor en ese instante.

## Escucha ciega (registro de escucha)

1. Pestaña «Escucha ciega»: elegir A y B (ramas o revisiones recientes) y «Preparar escucha ciega».
2. El trabajo `blind` reutiliza `compare` (B igualada en sonoridad a A), calcula el puntaje del
   evaluador de cada versión (en caché por la clave de su render), sortea el orden con una semilla
   (`random.Random(seed)`; la API acepta `seed` para reproducirlo) y escribe `X.wav`/`Y.wav` en
   `workspace/listening/<proyecto>/<sesión>/`. Su resultado solo contiene la sesión y las URL de X e
   Y: ni la ruta ni el nombre revelan cuál es cuál.
3. El usuario escucha (conmutación sincronizada como en A/B), marca preferencia (X, Y o ninguna),
   califica cada versión de 1 a 5 y deja una nota opcional. «Guardar y revelar» envía el veredicto
   en términos X/Y; el servidor lo traduce a A/B con el orden guardado y muestra la identidad.

Los veredictos se guardan en `<proyecto>/listening.db` (SQLite, `storage/listening.py`): revisiones
A y B, etiquetas, perfil, claves de render, puntajes del evaluador en ese momento, semilla y orden,
preferencia, calificaciones, nota y fecha. Una sesión admite un solo veredicto.

- `GET /api/projects/{nombre}/listening` → `{verdicts, summary}` (más recientes primero).
- `POST /api/projects/{nombre}/listening` con `{session, preferred: x|y|none, rating_x, rating_y, note}`.
- `osky-remix listening <proyecto> [--json]`: tabla de veredictos y **acuerdo con el evaluador**:
  de los pares con preferencia y puntajes distintos, % en que la versión preferida también tiene el
  menor puntaje (el puntaje es una suma de severidades: menor = mejor). Sin preferencia, empates de
  puntaje y pares sin puntaje se informan aparte y no entran en el porcentaje; también por perfil.

`osky-remix gc` puede borrar el audio de sesiones viejas (`workspace/listening/`); el registro queda.

## Pruebas

`tests/integration/test_api.py` (API) y `tests/ui/test_ui.py` (Playwright: el navegador bloquea,
mueve faders y clips, edita notas, renderiza y compara; se verifica el efecto en el historial).
`tests/audio/test_visualize.py` contrasta la visualización con señales sintéticas (escalón de -20 a
-10 LUFS, mono/invertido, tono en su banda); `tests/integration/test_listening.py` cubre el sorteo
con semilla, la traducción X/Y → A/B, la estadística de acuerdo y el viaje completo por la API y la
CLI. La prueba de interfaz completa una escucha ciega y verifica el veredicto en el registro.


## Control seguro de revisiones

El historial indica la rama que se revertirá. La reversión exige que su revisión siga siendo
la mostrada; si hubo una edición concurrente, hay que recargar. Render y evaluación fijan la
revisión visible y el perfil solicitado. La reproducción de un render anterior se oculta al
cambiar el contenido del proyecto. Los trabajos de render/evaluación se pueden cancelar desde
la interfaz; la API también ofrece `POST /api/jobs/{id}/cancel`.
