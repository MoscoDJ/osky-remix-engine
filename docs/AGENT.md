# Automatización con Claude (etapa 5)

```bash
uv run osky-remix evaluate <proyecto> [--rev X]                 # mide y lista hallazgos
uv run osky-remix improve  <proyecto> --branch B [--max-iter 4]  # bucle del agente
uv run osky-remix vary     <proyecto> --from REV [--variants A,B,C]
```

## Evaluador (`agents/evaluator.py`)

"Escucha" mediante mediciones del master final y de las salidas post-fader:

| Hallazgo | Medición | Objetivo (política) |
|---|---|---|
| `energy_contrast` | LUFS del master por sección (marcadores) contra la de mayor energía | ≥ 1.5 LU por nivel de energía (intro 1, voz 2, drop 3) |
| `vocal_level` | voz − resto en 300 Hz–5 kHz, secciones con voz | −4 … +3 LU |
| `bass_balance` | bajo − bombo en la sección de mayor energía | −6 … 0 LU |
| `quality_*` | verificaciones FAIL del reporte de calidad | ninguna |

Puntaje = suma de severidades (0 = sin hallazgos). La energía de cada sección viene de
`meta.arrangement` (o del nombre, si el proyecto es anterior). Nota: comparar la voz contra
todo el espectro daba −12 LU en una mezcla donde se oye bien (el bombo y el bajo dominan la
sonoridad pero no la tapan); por eso se mide en la banda donde compiten.

## Acciones y políticas (`agents/actions.py`, `agents/policies.py`)

- `energy_contrast`: carril `auto-master-energy` de ganancia por sección en el master, antes del
  mastering. Solo recorta (no empuja el limitador); máx. 4 dB por iteración y 9 dB acumulados.
- `loudness_tradeoff`: si falta contraste y el limitador quita ≥ 3 dB de cresta, baja un peldaño
  en la escalera de perfiles (`melodic_techno_club` → `melodic_techno_dynamic` → `streaming`).
  Cambia volumen por dinámica; se puede desactivar (`allow_loudness_tradeoff`).
- `vocal_level` / `bass_balance`: ganancia de la pista, máx. 4 dB por iteración.
- **Nunca** sobre pistas bloqueadas: el hallazgo se reporta como "requiere al usuario".

## Planificador (`agents/planner.py`)

Por iteración: acciones → proyecto candidato con `apply_ops` (mismas validaciones del contrato)
→ render en memoria → evaluación. Si el puntaje mejora ≥ 0.25 se confirma como revisión (autor
`claude`, mensaje con el motivo y el puntaje); si no, se descarta sin dejar rastro y el bucle termina.

## Variaciones (`composition/arranger.py: rearrange`)

Mismas secciones (la voz no se mueve), distinto contenido:

| Variante | Cambios |
|---|---|
| A | rodante (base) |
| B | arpegio en octavas, lead con motivo propio y registro alto, bajo a contratiempo en la voz, intro con más batería |
| C | arpegio pulsante, lead una octava más arriba, bajo sostenido en intro y voz, batería más ligera en la voz, pad en el drop |

Solo se regeneran pistas generadas y **no bloqueadas**; las bloqueadas se conservan tal cual y se
informa. `vary` reconstruye el contexto desde `meta.compose` (análisis, voz, inicio, compases,
tempo, semilla, plantilla).

## Evidencia con la canción real ("Tu Última Canción", 32 compases)

Hito 1 + etapa 4 (`hito1-mix2`): intro −8.90, voz −8.63, drop −8.82 LUFS (energía plana, LRA 0.49 LU).

`improve` en `hito1-energia`:

| It. | Acción | Puntaje |
|---|---|---|
| 1 | intro −3.8 dB, voz −2.3 dB | 4.61 → 2.49 ✓ |
| 2 | intro −5.8, voz −3.9 | 2.49 → 1.86 ✓ |
| 3 | intro −7.3, voz −5.3 | 1.86 → 1.37 ✓ |
| 4 | intro −8.3, voz −6.5 | 1.37 → 1.42 ✗ descartado |
| 5 | perfil → `melodic_techno_dynamic` (el limitador quitaba ~10 dB) | 1.37 → 0.00 ✓ |

Resultado: intro −12.9, voz −10.3, drop −8.6 LUFS; LRA 4.5 LU; master −10.3 LUFS.

Hito 2: el usuario sube la voz +1 dB y baja el bajo −1 dB y bloquea ambas pistas; `vary` crea
`var-a`, `var-b`, `var-c`. Voz y bajo idénticos al modelo de `manual` en las tres; entre variantes
cambian 58–576 notas por pista; las tres conservan el contraste. `improve` sobre `var-c` no cambia
nada y reporta los dos hallazgos de las pistas bloqueadas como decisión del usuario.
