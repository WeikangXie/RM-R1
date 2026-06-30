#!/usr/bin/env python3
"""Run label-conditioned second-pass review for LLM/operator disagreements."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from common import (
    VALID_LABELS,
    build_llm_url,
    extract_completion_content,
    extract_json_object,
    parse_rubrics,
    read_jsonl,
    read_rubrics,
    rubrics_text,
    text_or_empty,
    write_json,
    write_jsonl,
)


VALID_STATUSES = {"ok", "need_review"}


SYSTEM_PROMPT = """你是金融内容社区的 AI 回复审核复核助手。

运营最终审核结果已经给定。你的任务不是重新决定 pass/reject，而是严格依据 rubrics 判断该运营结果是否能被合理解释。

输出必须是 JSON，不要输出 JSON 之外的文字。JSON 字段固定为：
{
  "status": "ok 或 need_review",
  "violated_rubrics": ["命中的规则名称；如果没有则为空数组"],
  "reasoning": "引用具体内容依据，解释为什么给定运营结果成立；如果无法合理解释，说明为什么需要人工复核",
  "decision": "pass 或 reject，必须等于给定运营结果"
}

要求：
1. decision 必须等于给定运营结果。
2. 如果给定运营结果能被 rubrics 合理解释，status 输出 ok。
3. 如果无法根据当前 rubrics 和文本合理解释给定运营结果，status 输出 need_review，不要强行编造理由。
4. decision=pass 时，violated_rubrics 通常应为空数组，并解释为什么未命中违规规则。
5. decision=reject 时，violated_rubrics 应尽量给出命中的规则，并在 reasoning 中引用 AI 回复或上下文中的具体表达。
"""


def normalize_label(value: Any) -> str:
    return text_or_empty(value).strip().lower()


def is_disagreement(row: dict[str, Any]) -> bool:
    audit_label = normalize_label(row.get("audit_label"))
    llm_decision = normalize_label(row.get("llm_decision"))
    return audit_label in VALID_LABELS and llm_decision in VALID_LABELS and audit_label != llm_decision


def user_prompt(row: dict[str, Any], rubrics: list[dict[str, str]]) -> str:
    first_pass_rubrics = " | ".join(parse_rubrics(row.get("violated_rubrics")))
    return f"""请对下面这条金融内容社区 AI 回复做二次复核推理。

Rubrics:
{rubrics_text(rubrics)}

运营最终审核结果：
{normalize_label(row.get("audit_label"))}

第一次 LLM 审核结果：
- decision: {normalize_label(row.get("llm_decision"))}
- violated_rubrics: {first_pass_rubrics}
- reasoning: {text_or_empty(row.get("llm_reasoning"))}

业务上下文：
- 原帖/父评论：{text_or_empty(row.get("text"))}
- 产品名称：{text_or_empty(row.get("product_name"))}
- 话题：{text_or_empty(row.get("topic_titles"))}
- 圈子：{text_or_empty(row.get("coterie_names"))}
- 回复类型：{text_or_empty(row.get("extend_type"))}

待审核 AI 回复：
{text_or_empty(row.get("ai_reply"))}

请判断“运营最终审核结果”是否能被上面的 rubrics 合理解释。能解释则 status=ok，并给出与运营结果一致的 violated_rubrics 和 reasoning；不能解释则 status=need_review，并说明需要人工复核的原因。
"""


def normalize_annotation(annotation: dict[str, Any], expected_decision: str, valid_rubrics: set[str]) -> dict[str, Any]:
    status = text_or_empty(annotation.get("status")).strip().lower()
    if status not in VALID_STATUSES:
        raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}, got {status!r}")

    decision = normalize_label(annotation.get("decision"))
    if decision != expected_decision:
        raise ValueError(f"decision must equal audit_label {expected_decision!r}, got {decision!r}")

    violated_rubrics = annotation.get("violated_rubrics", [])
    if not isinstance(violated_rubrics, list):
        raise ValueError("violated_rubrics must be a list")
    normalized_rubrics = [str(item).strip() for item in violated_rubrics if str(item).strip()]
    unknown_rubrics = [item for item in normalized_rubrics if item not in valid_rubrics]
    if unknown_rubrics:
        raise ValueError(f"unknown rubrics: {unknown_rubrics}")

    reasoning = text_or_empty(annotation.get("reasoning")).strip()
    if not reasoning:
        raise ValueError("reasoning must not be empty")

    return {
        "status": status,
        "violated_rubrics": normalized_rubrics,
        "reasoning": reasoning,
        "decision": decision,
    }


def call_llm(row: dict[str, Any], rubrics: list[dict[str, str]], valid_rubrics: set[str], args: argparse.Namespace) -> dict[str, Any]:
    sample_id = text_or_empty(row.get("sample_id"))
    audit_label = normalize_label(row.get("audit_label"))
    payload: dict[str, Any] = {
        "model": args.llm_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(row, rubrics)},
        ],
        "max_tokens": args.llm_max_tokens,
        "temperature": args.llm_temperature,
        "top_p": args.llm_top_p,
        "stream": False,
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
            annotation = normalize_annotation(extract_json_object(raw_content), audit_label, valid_rubrics)
            return {
                "custom_id": sample_id,
                "audit_label": audit_label,
                "llm_decision": normalize_label(row.get("llm_decision")),
                "ok": True,
                "annotation": annotation,
                "raw_content": raw_content,
            }
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < args.llm_retries:
                time.sleep(args.llm_retry_sleep)

    if args.llm_fail_fast:
        raise RuntimeError(f"Second-pass annotation failed for {sample_id}: {last_error}") from last_error
    return {
        "custom_id": sample_id,
        "audit_label": audit_label,
        "llm_decision": normalize_label(row.get("llm_decision")),
        "ok": False,
        "annotation": None,
        "error": str(last_error),
    }


def make_summary(rows: list[dict[str, Any]], disagreements: list[dict[str, Any]], results: list[dict[str, Any]] | None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total_rows": len(rows),
        "selected_disagreements": len(disagreements),
        "audit_label_counts": dict(Counter(normalize_label(row.get("audit_label")) for row in rows)),
        "llm_decision_counts": dict(Counter(normalize_label(row.get("llm_decision")) for row in rows)),
        "disagreement_label_pairs": dict(
            Counter(f"{normalize_label(row.get('llm_decision'))}->{normalize_label(row.get('audit_label'))}" for row in disagreements)
        ),
    }
    if results is not None:
        summary["second_pass_counts"] = dict(Counter("ok" if result.get("ok") else "failed" for result in results))
        summary["second_pass_status_counts"] = dict(
            Counter(
                result["annotation"]["status"]
                for result in results
                if result.get("ok") and isinstance(result.get("annotation"), dict)
            )
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run second-pass review for rows where llm_decision disagrees with audit_label.")
    parser.add_argument("--human-review", type=Path, required=True, help="Input human_review.jsonl with first-pass LLM fields.")
    parser.add_argument("--rubrics", type=Path, required=True, help="Rubrics markdown file.")
    parser.add_argument("--output", type=Path, required=True, help="Output second-pass annotations JSONL.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of disagreement rows to process.")
    parser.add_argument("--dry-run", action="store_true", help="Only count selected disagreement rows; do not call LLM.")
    parser.add_argument("--llm-base-url", default=os.getenv("LLM_BASE_URL", ""), help="Base URL, e.g. http://host:port")
    parser.add_argument("--llm-model", default=os.getenv("LLM_MODEL", ""), help="Model name used in path and request body.")
    parser.add_argument(
        "--llm-authorization",
        default=os.getenv("LLM_AUTHORIZATION", ""),
        help="Authorization header value, e.g. Bearer xxx.",
    )
    parser.add_argument("--llm-max-tokens", type=int, default=768)
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-top-p", type=float, default=1.0)
    parser.add_argument("--llm-response-format", default="json_object", choices=["json_object", "text"])
    parser.add_argument("--llm-seed", type=int, default=None)
    parser.add_argument("--llm-timeout", type=float, default=60.0)
    parser.add_argument("--llm-retries", type=int, default=2)
    parser.add_argument("--llm-retry-sleep", type=float, default=1.0)
    parser.add_argument("--llm-fail-fast", action="store_true", help="Stop on the first failed second-pass annotation.")
    args = parser.parse_args()
    if not args.dry_run and (not args.llm_base_url or not args.llm_model or not args.llm_authorization):
        parser.error("LLM calls require --llm-base-url, --llm-model, and --llm-authorization, or matching env vars. Use --dry-run to only inspect disagreements.")
    return args


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.human_review)
    rubrics = read_rubrics(args.rubrics)
    valid_rubrics = {rubric["name"] for rubric in rubrics}

    disagreements = [row for row in rows if is_disagreement(row)]
    if args.limit is not None:
        disagreements = disagreements[: args.limit]

    summary_path = args.output.parent / "second_pass_summary.json"
    if args.dry_run:
        summary = make_summary(rows, disagreements, results=None)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    results: list[dict[str, Any]] = []
    for index, row in enumerate(disagreements, start=1):
        sample_id = text_or_empty(row.get("sample_id"))
        print(f"Second-pass annotating {index}/{len(disagreements)}: {sample_id}")
        results.append(call_llm(row, rubrics, valid_rubrics, args))
        write_jsonl(args.output, results)

    summary = make_summary(rows, disagreements, results)
    summary["outputs"] = {
        "second_pass_annotations": str(args.output),
        "summary": str(summary_path),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
