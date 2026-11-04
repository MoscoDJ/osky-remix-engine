"""Políticas del agente: qué puede cambiar, cuánto por iteración y qué objetivos persigue."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Policy:
    author: str = "claude"
    max_iterations: int = 4
    max_gain_step_db: float = 4.0            # cambio máximo de ganancia por acción e iteración
    max_total_section_cut_db: float = 9.0    # recorte acumulado máximo de una sección (energía)
    min_energy_delta_lu: float = 1.5         # LU mínimos por nivel de energía entre secciones
    vocal_window_lu: tuple[float, float] = (-4.0, 3.0)   # voz − resto en 300 Hz-5 kHz, secciones con voz
    bass_kick_window_lu: tuple[float, float] = (-6.0, 0.0)
    min_improvement: float = 0.25            # mejora mínima de puntaje para aceptar un cambio
    touch_locked: bool = False               # nunca: las pistas bloqueadas son del usuario
    # cambiar volumen por dinámica: si falta contraste y el limitador trabaja mucho, bajar un peldaño
    allow_loudness_tradeoff: bool = True
    profile_ladder: tuple[str, ...] = ("melodic_techno_club", "melodic_techno_dynamic", "streaming")
    limiting_threshold_db: float = 3.0       # reducción estimada (premaster→master) que justifica bajar
    cut_before_tradeoff_db: float = 6.0      # recorte por sección que se intenta antes de ceder volumen
    # enmascaramiento (analysis/masking.py) y EQ/ducking correctivos (agents/mix_actions.py)
    masking_threshold_db: float = -3.0       # enmascarador − objetivo en la banda, cuadros activos del objetivo
    max_eq_step_db: float = 3.0              # cambio máximo de una banda de EQ por iteración
    max_eq_cut_db: float = 9.0               # recorte acumulado máximo de una banda en un enmascarador
    max_eq_actions: int = 2                  # bandas de EQ que se tocan por iteración (por tipo de hallazgo)
    duck_step: float = 0.15                  # aumento de profundidad del sidechain por iteración
    max_duck_depth: float = 0.85
    # referencia comercial (analysis/reference_match.py), solo si meta.reference_track
    reference_band_tol_db: float = 2.0       # Δ tonal por banda de octava (alineado por la mediana)
    reference_width_tol_db: float = 4.0      # Δ lado/centro por banda
    reference_lra_tol_lu: float = 3.0
    reference_plr_tol_db: float = 2.5
    reference_punch_tol_db: float = 3.0
    max_reference_eq_db: float = 6.0         # ganancia acumulada máxima por banda en el EQ del master


DEFAULT = Policy()
