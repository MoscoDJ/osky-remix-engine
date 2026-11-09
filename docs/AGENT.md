# Automatización con Claude (etapa 5)

```bash
uv run osky-remix evaluate <proyecto> [--rev X]                 # mide y lista hallazgos
uv run osky-remix improve  <proyecto> --branch B [--max-iter 4]  # bucle del agente
uv run osky-remix vary     <proyecto> --from REV [--variants A,B,C]
uv run osky-remix match    <proyecto> [--rev X] --reference ref.wav [--set] [--json]
```

## Evaluador (`agents/evaluator.py`)

"Escucha" mediante mediciones del master final y de las salidas post-fader:

| Hallazgo | Medición | Objetivo (política) |
|---|---|---|
| `energy_contrast` | LUFS del master por sección (marcadores) contra la de mayor energía | ≥ 1.5 LU por nivel de energía (intro 1, voz 2, drop 3) |
| `vocal_level` | voz − resto en 300 Hz–5 kHz, secciones con voz | −4 … +3 LU |
| `bass_balance` | bajo − bombo en la sección de mayor energía | −6 … 0 LU |
| `masking` | enmascarador − objetivo en una banda de octava, mediana sobre los cuadros activos del objetivo, por sección (voz vs pad/lead/arp/chops; bajo → bombo) | ≤ −3 dB (`masking_threshold_db`); severidad 0.1/dB |
| `reference_tonal` | Δ tonal referencia − mezcla por banda de octava, alineado por la mediana (solo con `meta.reference_track`) | ±2 dB |
| `reference_width` | Δ lado/centro por banda | ±4 dB |
| `reference_dynamics` | Δ LRA / Δ PLR (true peak − LUFS) | ±3 LU / ±2.5 dB |
| `reference_punch` | Δ pico/RMS de la envolvente de 10 ms en 40–120 Hz | ±3 dB |
| `quality_*` | verificaciones FAIL del reporte de calidad | ninguna |

Puntaje = suma de severidades (0 = sin hallazgos). La energía de cada sección viene de
`meta.arrangement` (o del nombre, si el proyecto es anterior). Nota: comparar la voz contra
todo el espectro daba −12 LU en una mezcla donde se oye bien (el bombo y el bajo dominan la
sonoridad pero no la tapan); por eso se mide en la banda donde compiten.

## Acciones y políticas (`agents/actions.py`, `agents/policies.py`)

- `energy_contrast`: carril `auto-master-energy` de ganancia por sección en el master, antes del
  mastering. Solo recorta (no empuja el limitador); máx. 4 dB por iteración y 9 dB acumulados.
- `drum_clip`: si falta contraste y el limitador quita ≥ 3 dB de cresta, recorte suave
  (`clipper`, lineal hasta media altura, `tanh` encima, sobremuestreo 4x) al inicio del bus de
  batería: 2 dB de pico por iteración, hasta 6 dB, con el techo relativo al pico de entrada medido
  del bus. Los bombos de sample tienen mucha más cresta que el integrado (que ya satura) y son
  los que obligan al limitador del master a trabajar.
- `loudness_tradeoff`: si falta contraste y el limitador quita ≥ 3 dB de cresta, baja un peldaño
  en la escalera de perfiles (`melodic_techno_club` → `melodic_techno_dynamic`). `streaming`
  (−14 LUFS) quedó fuera: es una decisión de entrega, no de mezcla (las plataformas normalizan),
  y un juego de sonidos (MT3) terminaba 4 LU más bajo que los demás; se pide al exportar.
  Cambia volumen por dinámica; se puede desactivar (`allow_loudness_tradeoff`). Exige una
  mejora de puntaje ≥ 1.0 (`tradeoff_min_improvement`): con MT3 el agente bajó a `streaming`
  (−14 LUFS) para borrar un residuo de 0.30.
- Orden de la familia de energía (una por iteración): recortar las secciones con déficit
  mientras alguna de ellas lleve menos de `cut_before_tradeoff_db` (6 dB); después `drum_clip`,
  `loudness_tradeoff` y, por último, seguir recortando hasta 9 dB. El criterio mira los recortes
  de las secciones *con déficit*: con el máximo de cualquier sección (un intro a −7 dB) el agente
  cedía volumen mientras las secciones planas aún tenían margen.
- Alternativas: si la propuesta no mejora, se descarta sin rastro y, si incluía una acción de la
  familia de energía, se prueba la siguiente en la misma iteración (`max_alternatives` = 2).
- `vocal_level` / `bass_balance`: ganancia de la pista, máx. 4 dB por iteración.
- `masking` (voz): campana del EQ integrado (`eq`, Q √2) en el enmascarador, en el centro de la
  banda en conflicto (slot `auto-unmask-<banda>`); máx. 3 dB por iteración y 9 dB acumulados.
  Hasta 2 bandas por iteración, la peor primero.
- `masking` (bajo → bombo): más sidechain del bajo desde el bombo (efecto `sidechain` +
  `sidechain_from`; crea `auto-duck` si no había), +0.15 por iteración hasta 0.85; agotado, EQ.
- `reference_tonal`: campana en el master antes del mastering (slot `auto-ref-eq-<banda>`), máx.
  3 dB por iteración y ±6 dB acumulados, hasta 2 bandas por iteración. `reference_width`,
  `reference_dynamics` y `reference_punch` solo se informan (no hay acción automática).
- **Nunca** sobre pistas bloqueadas: el hallazgo se reporta como "requiere al usuario" (también
  cuando un recorte o EQ llegó a su límite).

## Comparar con una referencia (`analysis/reference_match.py`, `match`)

`match` renderiza la revisión con la caché de renders (`render_cached`), importa la referencia
(cualquier formato y frecuencia de muestreo; se remuestrea a la del proyecto) al almacén como asset
`reference` y compara el master con ella. Con `--set` la registra en el proyecto (asset +
`meta.reference_track = {sha256, name}`) y desde entonces `evaluate`/`improve` añaden los
hallazgos `reference_*`. El reporte se guarda junto al render (`match_<sha>.json`).

Métricas (todas independientes del nivel; además la referencia se iguala a la sonoridad de la
mezcla, `ref_gain_to_match_db`):

- **Balance tonal**: espectro promedio a largo plazo (Welch, Hann 8192) integrado en 10 bandas de
  octava ISO (31.5 Hz … 16 kHz, bordes fc/√2 … fc·√2). Δ = referencia − mezcla, menos la mediana de
  los Δ: igualar por LUFS desplazaría todas las bandas cuando difiere una sola (la ponderación K
  pesa más los agudos). Positivo = a la mezcla le falta ("faltan ~3 dB entre 5.7–11.3 kHz").
  También por sección (marcadores) contra la referencia completa, solo informativo.
- **Estéreo por banda**: lado/centro (M=(L+R)/2, S=(L−R)/2) y correlación L/R en la banda (parte
  real del espectro cruzado sobre √(P_LL·P_RR)).
- **Dinámica**: LRA (aprox. EBU), cresta (pico de muestra / RMS) y PLR (true peak − LUFS).
- **Pegada en graves** (40–120 Hz): percentil 99 de la envolvente RMS de 10 ms menos el RMS de la
  banda, y ataques por segundo (subida ≥ 6 dB en ≤ 30 ms).

Verificado con ruido de espectro exacto: rosa y −3 dB/oct dan la energía analítica por banda con
error < 0.5 dB; una referencia a 44.1 kHz con +6 dB en la octava de 8 kHz da un único hallazgo
(+6.0 dB en esa banda, el resto < 0.5 dB); mono → correlación 1, canales independientes → 0,
L=n₁, R=0.6n₁+0.8n₂ → 0.6 y lado/centro −6 dB.

Limitaciones: la referencia se compara entera (no por secciones propias); el EQ correctivo es una
campana estática (el `eq` integrado no es automatizable), por eso un recorte contra el
enmascaramiento afecta a todas las secciones aunque el conflicto esté en una sola.

## Enmascaramiento (`analysis/masking.py`)

Por par (objetivo, enmascarador), sección y banda de octava, con las salidas post-fader: energía
por cuadro (STFT 4096/2048); la banda cuenta si el objetivo tiene allí contenido (percentil 95 a
≤ 20 dB de su banda más fuerte); cuadros activos = a ≤ 15 dB de ese percentil 95 (los golpes del
bombo, las frases de la voz); `masking_db` = mediana de (enmascarador − objetivo) en esos cuadros.
Los pares de voz solo se miden en secciones con voz. `metrics.masking` lista todos los conflictos,
incluido bombo → bajo (solo informativo). Verificado: un pad de 1 kHz sobre una "voz" de
300 Hz–3 kHz se marca solo en la octava de 1 kHz (+8.9 dB) y un recorte de 3 dB lo baja
exactamente 3 dB; un lead en 8 kHz no se marca.

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
