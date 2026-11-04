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

## Pruebas

`tests/integration/test_api.py` (API) y `tests/ui/test_ui.py` (Playwright: el navegador bloquea,
mueve faders y clips, edita notas, renderiza y compara; se verifica el efecto en el historial).
