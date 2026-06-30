#!/usr/bin/env python3
"""Prepare first-pass business-domain RM data artifacts.

The script keeps the data stage deliberately small:
raw company JSONL -> optional LLM annotations -> human review JSONL.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from common import (
    build_llm_url,
    extract_completion_content,
    extract_json_object,
    read_jsonl,
    read_rubrics,
    rubrics_text,
    text_or_empty,
    write_json,
    write_jsonl,
)
from config import (
    FIRST_PASS_LLM_MAX_TOKENS,
    LLM_AUTHORIZATION,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_RESPONSE_FORMAT,
    LLM_RETRIES,
    LLM_RETRY_SLEEP,
    LLM_SEED,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    LLM_TOP_P,
)


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


def call_llm(task: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "model": LLM_MODEL,
        "messages": task["messages"],
        "max_tokens": FIRST_PASS_LLM_MAX_TOKENS,
        "temperature": LLM_TEMPERATURE,
        "top_p": LLM_TOP_P,
        "stream": False,
        # The company interface documents response_format as json_object/text.
        # json_object makes annotation parsing stricter and downstream data clean.
        "response_format": LLM_RESPONSE_FORMAT,
    }
    if LLM_SEED is not None:
        payload["seed"] = LLM_SEED

    request = urllib.request.Request(
        build_llm_url(LLM_BASE_URL, LLM_MODEL),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": LLM_AUTHORIZATION,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(1, LLM_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=LLM_TIMEOUT) as response:
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
            if attempt < LLM_RETRIES:
                time.sleep(LLM_RETRY_SLEEP)

    return {
        "custom_id": task["custom_id"],
        "audit_label": task["audit_label"],
        "ok": False,
        "annotation": None,
        "error": str(last_error),
    }


def run_llm_annotations(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not LLM_AUTHORIZATION:
        raise ValueError("LLM authorization is required. Set LLM_AUTHORIZATION in content_rm/data/config.py or the environment.")

    results: list[dict[str, Any]] = []
    for index, task in enumerate(tasks, start=1):
        print(f"Annotating {index}/{len(tasks)}: {task['custom_id']}")
        results.append(call_llm(task))
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare domain content RM data artifacts.")
    parser.add_argument("--input", type=Path, required=True, help="Raw company comment JSONL.")
    parser.add_argument("--rubrics", type=Path, required=True, help="Rubrics markdown file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for generated artifacts.")
    parser.add_argument("--write-normalized", action="store_true", help="Write normalized_samples.jsonl for debugging.")
    parser.add_argument("--call-llm", action="store_true", help="Call company-internal LLM and fill annotations.")
    args = parser.parse_args()
    if args.call_llm and (not LLM_BASE_URL or not LLM_MODEL):
        parser.error("--call-llm requires LLM_BASE_URL and LLM_MODEL in content_rm/data/config.py.")
    return args


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    rubrics = read_rubrics(args.rubrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = [make_sample(raw, index) for index, raw in enumerate(rows, start=1)]
    llm_tasks = [make_llm_task(sample, rubrics) for sample in samples]
    llm_results = run_llm_annotations(llm_tasks) if args.call_llm else []
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
    write_json(paths["summary"], summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
