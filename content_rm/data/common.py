#!/usr/bin/env python3
"""Shared helpers for content RM data scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


VALID_LABELS = {"pass", "reject"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no} must be a JSON object")
            rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def text_or_empty(value: Any) -> str:
    return value if isinstance(value, str) else ""


def read_rubrics(path: Path) -> list[dict[str, str]]:
    rubrics: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
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
    return "\n".join(f"- {r['name']}: {r['description']}" for r in rubrics)


def parse_rubrics(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if not isinstance(value, str) or not value.strip():
        return []
    if "|" in value:
        parts = value.split("|")
    elif "," in value:
        parts = value.split(",")
    else:
        parts = [value]
    return [part.strip() for part in parts if part.strip()]


def extract_json_object(text: str) -> dict[str, Any]:
    content = text.strip()
    if content.startswith("```"):
        lines = [line for line in content.splitlines() if not line.strip().startswith("```")]
        content = "\n".join(lines).strip()
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("LLM response does not contain a JSON object")
    parsed = json.loads(content[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON must be an object")
    return parsed


def extract_completion_content(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(first.get("text"), str):
                return first["text"]
    if isinstance(response.get("content"), str):
        return response["content"]
    raise ValueError("Cannot find completion text in LLM response")


def build_llm_url(base_url: str, model: str) -> str:
    return f"{base_url.rstrip('/')}/llm/{model}/v1/chat/completions"
