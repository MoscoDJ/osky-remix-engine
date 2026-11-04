"""Los esquemas publicados en schemas/ deben coincidir con el modelo (regenerar con
`uv run osky-remix schema export`)."""
import json

from pydantic import TypeAdapter

from osky_remix.core.config import REPO_ROOT
from osky_remix.core.models import Project
from osky_remix.core.operations import Operation


def test_project_schema_is_current():
    on_disk = json.loads((REPO_ROOT / "schemas" / "project.schema.json").read_text())
    assert on_disk == json.loads(json.dumps(Project.model_json_schema()))


def test_operation_schema_is_current():
    on_disk = json.loads((REPO_ROOT / "schemas" / "operation.schema.json").read_text())
    assert on_disk == json.loads(json.dumps(TypeAdapter(list[Operation]).json_schema()))


def test_render_schema_is_current():
    from osky_remix.audio.renderer import RenderReport
    on_disk = json.loads((REPO_ROOT / "schemas" / "render.schema.json").read_text())
    assert on_disk == json.loads(json.dumps(TypeAdapter(RenderReport).json_schema()))
