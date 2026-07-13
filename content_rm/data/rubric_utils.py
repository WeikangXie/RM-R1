"""Content RM rubric parsing and prompt formatting."""

from __future__ import annotations

from pathlib import Path


def read_rubrics(path: Path) -> list[dict[str, str]]:
    rubrics: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line.startswith("|"):
                continue
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) < 2:
                continue
            name, description = cells[0], cells[1]
            if name in {"Rubric", "-----------"} or set(name) <= {"-"}:
                continue
            rubrics.append({"name": name, "description": description})
    if not rubrics:
        raise ValueError(f"No rubrics parsed from {path}")
    return rubrics


def rubrics_text(rubrics: list[dict[str, str]]) -> str:
    return "\n".join(
        f"- {rubric['name']}: {rubric['description']}" for rubric in rubrics
    )
