#!/usr/bin/env python3
"""Prepare first-pass business-domain RM data artifacts.

The script keeps the data stage deliberately small:
raw company JSONL -> optional LLM annotations -> human review JSONL.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


LABEL_BY_COMMENT_STATE = {
    "PUBLISHED": "pass",
    "HIDE": "reject",
}


SYSTEM_PROMPT = """你是金融内容社区的 AI 回复审核助手。请严格依据给定 rubrics 判断 AI 回复是否可以通过运营审核。

输出必须是 JSON，不要输出 JSON 之外的文字。JSON 字段固定为：
{
  "violated_rubrics": ["命中的规则名称；如果没有则为空数组"],
  "reasoning": "简短说明为什么通过或不通过，必须引用具体内容依据",
  "decision": "pass 或 reject"
}

注意：
1. decision 只能是 pass 或 reject。
2. 若 AI 回复包含收益承诺、暗示确定收益、诱导买卖、个性化投资建议、事实不确定却说得过满等风险，应倾向 reject。
"""


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


def text_or_empty(value: Any) -> str:
    return value if isinstance(value, str) else ""


def clean_text(value: Any) -> str:
    raw_text = html.unescape(text_or_empty(value)).strip()
    if not raw_text:
        return ""

    try:
        from bs4 import BeautifulSoup  # type: ignore
    except ImportError:
        # Fallback keeps this script usable in minimal company machines where
        # bs4 may not be installed yet. It is enough for simple tag removal.
        without_breaks = re.sub(r"<\s*br\s*/?\s*>", "\n", raw_text, flags=re.IGNORECASE)
        without_tags = re.sub(r"<[^>]+>", "", without_breaks)
        return " ".join(html.unescape(without_tags).split())

    soup = BeautifulSoup(raw_text, "html.parser")
    return " ".join(soup.get_text(separator=" ", strip=True).split())


def compact_named_list(items: Any, key: str) -> list[str]:
    if not isinstance(items, list):
        return []
    values: list[str] = []
    for item in items:
        if isinstance(item, dict) and item.get(key):
            values.append(str(item[key]))
    return values


def make_sample(raw: dict[str, Any], index: int) -> dict[str, Any]:
    comment_state = text_or_empty(raw.get("commentState"))
    audit_label = LABEL_BY_COMMENT_STATE.get(comment_state, "unknown")

    sample_id = text_or_empty(raw.get("commentId")) or f"sample-{index:06d}"
    return {
        "sample_id": sample_id,
        "source_context": {
            "text": clean_text(raw.get("text")),
            "parent_type": text_or_empty(raw.get("parentType")),
        },
        "ai_reply": text_or_empty(raw.get("commentContent")),
        "metadata": {
            "product_name": text_or_empty(raw.get("productName")),
            "extend_type": text_or_empty(raw.get("extendType")),
            "comment_type": text_or_empty(raw.get("commentType")),
            "comment_state": comment_state,
            "topic_titles": compact_named_list(raw.get("topicList"), "title"),
            "coterie_names": compact_named_list(raw.get("relatedCoterieList"), "coterieName"),
            "create_time": text_or_empty(raw.get("createTime")),
            "audit_time": text_or_empty(raw.get("auditTime")),
        },
        # The audit label is the operator outcome. LLM/human annotations may
        # later explain or correct it, but this field preserves the source label.
        "audit_label": audit_label,
        "annotation": {
            "violated_rubrics": [],
            "reasoning": "",
            "decision": audit_label if audit_label in {"pass", "reject"} else "",
        },
    }


def rubrics_text(rubrics: list[dict[str, str]]) -> str:
    return "\n".join(f"- {r['name']}: {r['description']}" for r in rubrics)


def user_prompt(sample: dict[str, Any], rubrics: list[dict[str, str]]) -> str:
    context = sample["source_context"]
    metadata = sample["metadata"]
    return f"""请审核下面这条金融内容社区 AI 回复。

Rubrics:
{rubrics_text(rubrics)}

业务上下文：
- 原帖/父评论：{context['text']}
- 产品名称：{metadata['product_name']}
- 话题：{", ".join(metadata['topic_titles'])}
- 圈子：{", ".join(metadata['coterie_names'])}
- 回复类型：{metadata['extend_type']}

待审核 AI 回复：
{sample['ai_reply']}
"""


def make_llm_task(sample: dict[str, Any], rubrics: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "custom_id": sample["sample_id"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(sample, rubrics)},
        ],
        "audit_label": sample["audit_label"],
    }


def normalize_annotation(annotation: dict[str, Any]) -> dict[str, Any]:
    violated_rubrics = annotation.get("violated_rubrics", [])
    if not isinstance(violated_rubrics, list):
        violated_rubrics = []

    decision = text_or_empty(annotation.get("decision")).strip().lower()
    if decision not in {"pass", "reject"}:
        raise ValueError(f"annotation decision must be pass or reject, got {decision!r}")

    return {
        "violated_rubrics": [str(item) for item in violated_rubrics],
        "reasoning": text_or_empty(annotation.get("reasoning")).strip(),
        "decision": decision,
    }


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
    """Read common OpenAI-compatible chat completion response shapes."""
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


def call_llm(task: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "model": args.llm_model,
        "messages": task["messages"],
        "max_tokens": args.llm_max_tokens,
        "temperature": args.llm_temperature,
        "top_p": args.llm_top_p,
        "stream": False,
        # The company interface documents response_format as json_object/text.
        # json_object makes annotation parsing stricter and downstream data clean.
        "response_format": args.llm_response_format,
    }
    if args.llm_seed is not None:
        payload["seed"] = args.llm_seed

    request = urllib.request.Request(
        build_llm_url(args.llm_base_url, args.llm_model),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": args.llm_authorization,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(1, args.llm_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=args.llm_timeout) as response:
                body = response.read().decode("utf-8")
            response_json = json.loads(body)
            raw_content = extract_completion_content(response_json)
            annotation = normalize_annotation(extract_json_object(raw_content))
            return {
                "custom_id": task["custom_id"],
                "audit_label": task["audit_label"],
                "ok": True,
                "annotation": annotation,
                "raw_content": raw_content,
            }
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < args.llm_retries:
                time.sleep(args.llm_retry_sleep)

    if args.llm_fail_fast:
        raise RuntimeError(f"LLM annotation failed for {task['custom_id']}: {last_error}") from last_error
    return {
        "custom_id": task["custom_id"],
        "audit_label": task["audit_label"],
        "ok": False,
        "annotation": None,
        "error": str(last_error),
    }


def run_llm_annotations(tasks: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.llm_authorization:
        raise ValueError("LLM authorization is required. Pass --llm-authorization or set LLM_AUTHORIZATION.")

    results: list[dict[str, Any]] = []
    for index, task in enumerate(tasks, start=1):
        print(f"Annotating {index}/{len(tasks)}: {task['custom_id']}")
        results.append(call_llm(task, args))
    return results


def make_human_review_row(sample: dict[str, Any], llm_annotation: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = sample["metadata"]
    context = sample["source_context"]
    annotation = llm_annotation or {}
    return {
        "sample_id": sample["sample_id"],
        "audit_label": sample["audit_label"],
        "llm_decision": text_or_empty(annotation.get("decision")),
        "llm_reasoning": text_or_empty(annotation.get("reasoning")),
        "human_reasoning": "",
        "review_note": "",
        "violated_rubrics": " | ".join(annotation.get("violated_rubrics", [])),
        "extend_type": metadata["extend_type"],
        "comment_state": metadata["comment_state"],
        "product_name": metadata["product_name"],
        "topic_titles": " | ".join(metadata["topic_titles"]),
        "coterie_names": " | ".join(metadata["coterie_names"]),
        "text": context["text"],
        "ai_reply": sample["ai_reply"],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare domain content RM data artifacts.")
    parser.add_argument("--input", type=Path, required=True, help="Raw company comment JSONL.")
    parser.add_argument("--rubrics", type=Path, required=True, help="Rubrics markdown file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for generated artifacts.")
    parser.add_argument("--write-normalized", action="store_true", help="Write normalized_samples.jsonl for debugging.")
    parser.add_argument("--call-llm", action="store_true", help="Call company-internal LLM and fill annotations.")
    parser.add_argument("--llm-base-url", default=os.getenv("LLM_BASE_URL", ""), help="Base URL, e.g. http://host:port")
    parser.add_argument("--llm-model", default=os.getenv("LLM_MODEL", ""), help="Model name used in path and request body.")
    parser.add_argument(
        "--llm-authorization",
        default=os.getenv("LLM_AUTHORIZATION", ""),
        help="Authorization header value, e.g. Bearer xxx.",
    )
    parser.add_argument("--llm-max-tokens", type=int, default=512)
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-top-p", type=float, default=1.0)
    parser.add_argument("--llm-response-format", default="json_object", choices=["json_object", "text"])
    parser.add_argument("--llm-seed", type=int, default=None)
    parser.add_argument("--llm-timeout", type=float, default=60.0)
    parser.add_argument("--llm-retries", type=int, default=2)
    parser.add_argument("--llm-retry-sleep", type=float, default=1.0)
    parser.add_argument("--llm-fail-fast", action="store_true", help="Stop on the first failed LLM annotation.")
    args = parser.parse_args()
    if args.call_llm and (not args.llm_base_url or not args.llm_model):
        parser.error("--call-llm requires --llm-base-url and --llm-model, or LLM_BASE_URL/LLM_MODEL env vars.")
    return args


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    rubrics = read_rubrics(args.rubrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = [make_sample(raw, index) for index, raw in enumerate(rows, start=1)]
    llm_tasks = [make_llm_task(sample, rubrics) for sample in samples]
    llm_results = run_llm_annotations(llm_tasks, args) if args.call_llm else []
    llm_annotations_by_id = {
        item["custom_id"]: item["annotation"]
        for item in llm_results
        if item.get("ok") and isinstance(item.get("annotation"), dict)
    }
    review_rows = [make_human_review_row(sample, llm_annotations_by_id.get(sample["sample_id"])) for sample in samples]

    paths = {
        "human_review_jsonl": args.output_dir / "human_review.jsonl",
        "summary": args.output_dir / "summary.json",
    }
    if args.write_normalized:
        paths["normalized_samples"] = args.output_dir / "normalized_samples.jsonl"
    if args.call_llm:
        paths["llm_annotations"] = args.output_dir / "llm_annotations.jsonl"

    if args.write_normalized:
        write_jsonl(paths["normalized_samples"], samples)
    if args.call_llm:
        write_jsonl(paths["llm_annotations"], llm_results)
    write_jsonl(paths["human_review_jsonl"], review_rows)

    summary = {
        "input": str(args.input),
        "rubrics": str(args.rubrics),
        "total": len(samples),
        "audit_label_counts": dict(Counter(sample["audit_label"] for sample in samples)),
        "comment_state_counts": dict(Counter(sample["metadata"]["comment_state"] for sample in samples)),
        "extend_type_counts": dict(Counter(sample["metadata"]["extend_type"] for sample in samples)),
        "llm_annotation_counts": dict(Counter("ok" if item.get("ok") else "failed" for item in llm_results)),
        "outputs": {name: str(path) for name, path in paths.items() if name != "summary"},
    }
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
