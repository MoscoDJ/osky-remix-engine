# Reglas para agentes (Claude)

- Etapas en orden (`docs/ROADMAP.md`). No avanzar si una prueba crítica de la etapa actual falla.
- Ninguna prueba se declara aprobada por código de salida 0: se verifica el audio o el
  archivo resultante contra valores conocidos de fixtures sintéticos.
- No relajar un criterio para que una prueba pase. Si un criterio era incorrecto, documentar
  por qué en `docs/COMPATIBILITY.md` en el mismo cambio.
- Posiciones musicales en ticks (PPQ 960) y de audio en muestras enteras; nunca solo segundos flotantes.
- El proyecto se modifica solo mediante operaciones validadas y atómicas (etapa 1 en adelante).
- La voz y todo asset original permanecen recuperables (no destructivo).
- Plugins de Surge XT: usar `audio.plugin_host` (pre-roll, presets .fxp, parámetros como
  automatización constante). Ver hallazgos en `docs/COMPATIBILITY.md`.
- Canciones, stems, modelos y renders van en `workspace/` (ignorado por git).
- Commits sin atribución a IA.
- Ejecutar `uv run pytest -m "not compat"` antes de dar un cambio por terminado; si se tocan
  audio, plugins o mastering, también `uv run osky-remix doctor`.
