# Flujo de trabajo de Claude con el motor

Claude opera el motor solo por la CLI y el contrato de operaciones; nunca edita archivos del
proyecto ni la base de datos.

1. **Entorno**: `osky-remix doctor` antes de producir; si una prueba crítica falla, se detiene.
2. **Material**: `project new` → `ingest` (original intacto + copia de trabajo) →
   `separate -b demucs --lead-backing` → `analyze --stems`.
3. **Leer el análisis antes de decidir**: tempo y tramos libres, tonalidad y modulaciones,
   secciones, dónde canta la voz. Confianza baja = verificar o preguntar, no suponer.
4. **Rama por idea**: `project branch <p> idea --from main`; nunca componer sobre `main`.
5. **Componer**: `compose --analysis … --vocals … --at m:ss --bars N --branch idea`.
6. **Escuchar con herramientas**: `render`, `evaluate`, `compare main idea`. Comparar siempre
   igualado en sonoridad.
7. **Corregir**: `improve --branch idea` o lotes propios con `op apply -a claude -m "motivo"`.
   Un lote es atómico; si falla, leer el índice y el motivo, no reintentar a ciegas.
8. **Variantes**: `vary --from idea` para comparar arreglos sin tocar lo bloqueado.
9. **Respetar al usuario**: pistas bloqueadas = ediciones del usuario. Claude no las desbloquea
   ni las altera; reporta los hallazgos que las involucran.
10. **Entregar**: `export <p> <carpeta> --rev X` y `package verify <carpeta>`.

El oído del usuario es la verificación final: las métricas guían, no sustituyen la escucha.
