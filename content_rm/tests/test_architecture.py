from __future__ import annotations

import ast
from pathlib import Path


def test_infrastructure_has_no_data_layer_imports() -> None:
    infrastructure_dir = Path(__file__).resolve().parents[1] / "infrastructure"
    forbidden_roots = {
        "app",
        "data",
        "records",
        "rubric_utils",
        "annotation_config",
        "prepare_dataset",
        "second_pass_review",
        "build_sft_dataset",
    }

    violations: list[str] = []
    for path in sorted(infrastructure_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported: list[str] = []
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported = [node.module]
            for module in imported:
                if module.split(".", maxsplit=1)[0] in forbidden_roots:
                    violations.append(f"{path.name}:{node.lineno} imports {module}")

    assert violations == []
