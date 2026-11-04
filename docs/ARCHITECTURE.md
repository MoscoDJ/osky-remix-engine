# Arquitectura

Tres responsabilidades separadas:

1. **Claude decide qué producir** (`agents/`): planifica y emite operaciones.
2. **El motor musical representa la producción** (`core/`, `composition/`, `analysis/`):
   proyecto canónico, timeline en ticks (PPQ 960) y muestras enteras, operaciones atómicas
   validadas con precondiciones y postcondiciones.
3. **El motor de audio ejecuta cómo suena** (`audio/`, `mastering/`, `separation/`):
   backends intercambiables (Surge XT vía DawDreamer, síntesis propia, Faust/Csound;
   Demucs/RoFormer; perfiles de mastering).

Claude no edita el archivo del proyecto directamente: toda modificación pasa por el contrato
de operaciones (`schemas/operation.schema.json`, etapa 1).

Los plugins y modelos se instalan por separado y se validan con `osky-remix doctor`. Si un
plugin falla, se registra, se aísla (cada prueba corre en su propio proceso) y se usa un
backend alternativo cuando exista.
