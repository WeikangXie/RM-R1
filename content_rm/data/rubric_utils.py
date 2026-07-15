"""Content RM rubric parsing and prompt formatting."""

from __future__ import annotations

from pathlib import Path


# Model outputs occasionally shorten a rubric title or vary punctuation. Keep
# these mappings explicit so normalization never guesses across business rules.
RUBRIC_NAME_ALIASES: dict[str, str] = {
    "不得诱导购买/卖出": "不得诱导购买 / 卖出",
    "诱导购买/卖出": "不得诱导购买 / 卖出",
    "诱导购买 / 卖出": "不得诱导购买 / 卖出",
    "不得诱导买卖": "不得诱导购买 / 卖出",
    "暗示收益": "不得暗示收益",
    "夸大产品表现": "不得夸大产品表现",
}


def canonicalize_rubric_names(
    values: list[str], valid_rubrics: set[str]
) -> list[str]:
    """Resolve known model-output aliases and reject genuinely unknown names."""

    canonical: list[str] = []
    seen: set[str] = set()
    unknown: list[str] = []
    for value in values:
        name = value.strip()
        if not name:
            continue
        if name in valid_rubrics:
            resolved = name
        else:
            resolved = RUBRIC_NAME_ALIASES.get(name, name)
        if resolved not in valid_rubrics:
            unknown.append(name)
            continue
        if resolved not in seen:
            canonical.append(resolved)
            seen.add(resolved)
    if unknown:
        raise ValueError(f"unknown rubrics: {unknown}")
    return canonical


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
