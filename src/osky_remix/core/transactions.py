"""Aplicación atómica de operaciones: todo el lote se aplica o el proyecto queda intacto."""
from __future__ import annotations

from pydantic import TypeAdapter, ValidationError

from .models import Project
from .operations import Context, Op, Operation, OperationError

_ops_adapter = TypeAdapter(list[Operation])


class TransactionError(Exception):
    def __init__(self, index: int, op: Op | dict, reason: str):
        self.index, self.op, self.reason = index, op, reason
        name = op.op if isinstance(op, Op) else op.get("op", "?")
        super().__init__(f"operación #{index} ({name}): {reason}")


def parse_ops(raw: list[dict]) -> list[Op]:
    """Valida un lote crudo (JSON) contra el contrato. Errores con índice de operación."""
    try:
        return _ops_adapter.validate_python(raw)
    except ValidationError as e:
        first = e.errors()[0]
        idx = first["loc"][0] if first["loc"] and isinstance(first["loc"][0], int) else -1
        op = raw[idx] if 0 <= idx < len(raw) else {}
        raise TransactionError(idx, op, f"no cumple el esquema: {first['msg']} en {first['loc'][1:]}") from None


def apply_ops(project: Project, ops: list[Op], ctx: Context) -> Project:
    """Devuelve un proyecto NUEVO con todas las operaciones aplicadas.

    Por operación: política de bloqueo -> precondiciones -> aplicar sobre copia ->
    revalidación completa del modelo -> postcondiciones. Ante cualquier fallo se lanza
    ``TransactionError`` y ``project`` no se modifica (nunca se muta el original).
    """
    current = project
    for i, op in enumerate(ops):
        try:
            locked = {t.id for t in current.tracks if t.locked}
            hit = op.touches(current) & locked
            if hit:
                raise OperationError(f"toca pistas bloqueadas: {', '.join(sorted(hit))}")
            op.check(current, ctx)
            draft = current.model_copy(deep=True)
            op.apply(draft)
            nxt = Project.model_validate(draft.model_dump())     # integridad completa
            op.verify(current, nxt)
        except TransactionError:
            raise
        except (OperationError, ValidationError, ValueError, KeyError) as e:
            msg = e.errors()[0]["msg"] if isinstance(e, ValidationError) else str(e)
            raise TransactionError(i, op, msg) from None
        current = nxt
    return current
