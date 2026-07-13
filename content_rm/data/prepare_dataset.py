#!/usr/bin/env python3
"""Prepare and annotate company content records for the Content RM workflow."""

from __future__ import annotations

import argparse
import html
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from pydantic import ValidationError

from async_pipeline import (
    append_retry_jobs,
    collect_latest_results,
    initialize_run,
    load_manifest,
    load_tasks,
    manifest_path,
    mark_failed_comments,
    plan_summary,
    refresh_status,
    status_summary,
    submit_run,
)
from cmb_async_model import CmbAsyncLLM
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
    LLM_RETRIES,
    LLM_RETRY_SLEEP,
    LLM_SEED,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    LLM_TOP_P,
)
from records import (
    AnnotationTask,
    AsyncRequestParams,
    FirstPassAnnotation,
    LLMAnnotationRecord,
    Message,
    RawComment,
    ReviewContext,
    model_rows,
)

LABEL_BY_COMMENT_STATE = {"PUBLISHED": "pass", "HIDE": "reject"}

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
    soup = BeautifulSoup(raw_text, "html.parser")
    return " ".join(soup.get_text(separator=" ", strip=True).split())


def make_context(raw: RawComment) -> ReviewContext:
    return ReviewContext(
        text=clean_text(raw.text),
        ai_reply=text_or_empty(raw.comment_content),
        product_name=text_or_empty(raw.product_name),
        extend_type=text_or_empty(raw.extend_type),
        comment_state=text_or_empty(raw.comment_state),
        topic_titles=[text_or_empty(item.title) for item in raw.topic_list if text_or_empty(item.title)],
        coterie_names=[
            text_or_empty(item.coterie_name)
            for item in raw.related_coterie_list
            if text_or_empty(item.coterie_name)
        ],
    )


def user_prompt(context: ReviewContext, rubrics: list[dict[str, str]]) -> str:
    return f"""请审核下面这条金融内容社区 AI 回复。

Rubrics:
{rubrics_text(rubrics)}

业务上下文：
- 原帖/父评论：{context.text}
- 产品名称：{context.product_name}
- 话题：{", ".join(context.topic_titles)}
- 圈子：{", ".join(context.coterie_names)}
- 回复类型：{context.extend_type}

待审核 AI 回复：
{context.ai_reply}
"""


def build_tasks(rows: list[dict[str, Any]], rubrics: list[dict[str, str]]) -> list[AnnotationTask]:
    tasks: list[AnnotationTask] = []
    seen: set[str] = set()
    for line_no, row in enumerate(rows, start=1):
        try:
            raw = RawComment.model_validate(row)
        except ValidationError as exc:
            raise ValueError(f"raw input line {line_no} is invalid: {exc}") from exc
        comment_id = str(raw.comment_id)
        if comment_id in seen:
            raise ValueError(f"duplicate comment_id in input: {comment_id}")
        seen.add(comment_id)
        comment_state = text_or_empty(raw.comment_state)
        audit_label = LABEL_BY_COMMENT_STATE.get(comment_state)
        if audit_label is None:
            raise ValueError(
                f"comment_id {comment_id} has unsupported commentState {comment_state!r}"
            )
        context = make_context(raw)
        tasks.append(
            AnnotationTask(
                comment_id=raw.comment_id,
                audit_label=audit_label,
                context=context,
                messages=[
                    Message(role="system", content=SYSTEM_PROMPT),
                    Message(role="user", content=user_prompt(context, rubrics)),
                ],
            )
        )
    return tasks


def validate_annotation(value: dict[str, Any], valid_rubrics: set[str]) -> FirstPassAnnotation:
    annotation = FirstPassAnnotation.model_validate(value)
    unknown = [item for item in annotation.violated_rubrics if item not in valid_rubrics]
    if unknown:
        raise ValueError(f"unknown rubrics: {unknown}")
    return annotation


def failed_record(task: AnnotationTask, error: str, raw_content: str | None = None) -> LLMAnnotationRecord:
    return LLMAnnotationRecord(
        comment_id=task.comment_id,
        audit_label=task.audit_label,
        context=task.context,
        ok=False,
        raw_content=raw_content,
        error=error,
    )


def parse_response_record(
    task: AnnotationTask,
    response: dict[str, Any],
    valid_rubrics: set[str],
) -> LLMAnnotationRecord:
    raw_content: str | None = None
    try:
        raw_content = extract_completion_content(response)
        annotation = validate_annotation(extract_json_object(raw_content), valid_rubrics)
        return LLMAnnotationRecord(
            comment_id=task.comment_id,
            audit_label=task.audit_label,
            context=task.context,
            ok=True,
            annotation=annotation,
            raw_content=raw_content,
        )
    except Exception as exc:
        return failed_record(task, str(exc), raw_content)


def sync_payload(task: AnnotationTask) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": LLM_MODEL,
        "messages": [message.model_dump() for message in task.messages],
        "max_tokens": FIRST_PASS_LLM_MAX_TOKENS,
        "temperature": LLM_TEMPERATURE,
        "top_p": LLM_TOP_P,
        "stream": False,
    }
    if LLM_SEED is not None:
        payload["seed"] = LLM_SEED
    return payload


def call_sync_llm(
    task: AnnotationTask,
    valid_rubrics: set[str],
    session: requests.Session,
) -> LLMAnnotationRecord:
    last_error: Exception | None = None
    last_failed_record: LLMAnnotationRecord | None = None
    for attempt in range(1, LLM_RETRIES + 1):
        try:
            response = session.post(
                build_llm_url(LLM_BASE_URL, LLM_MODEL),
                json=sync_payload(task),
                timeout=LLM_TIMEOUT,
            )
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict):
                raise ValueError("LLM response must be a JSON object")
            record = parse_response_record(task, body, valid_rubrics)
            if not record.ok:
                last_failed_record = record
                raise ValueError(record.error)
            return record
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < LLM_RETRIES:
                time.sleep(LLM_RETRY_SLEEP)
    return last_failed_record or failed_record(task, str(last_error))


def run_sync_annotations(
    tasks: list[AnnotationTask], valid_rubrics: set[str]
) -> list[LLMAnnotationRecord]:
    session = requests.Session()
    session.headers.update(
        {"Authorization": LLM_AUTHORIZATION, "Content-Type": "application/json"}
    )
    try:
        results: list[LLMAnnotationRecord] = []
        for index, task in enumerate(tasks, start=1):
            print(f"Annotating {index}/{len(tasks)}: {task.comment_id}")
            results.append(call_sync_llm(task, valid_rubrics, session))
        return results
    finally:
        session.close()


def make_async_client() -> CmbAsyncLLM:
    return CmbAsyncLLM(
        host=LLM_BASE_URL,
        authorization=LLM_AUTHORIZATION,
        model=LLM_MODEL,
        timeout=LLM_TIMEOUT,
        retries=LLM_RETRIES,
        retry_sleep=LLM_RETRY_SLEEP,
    )


def request_params() -> AsyncRequestParams:
    return AsyncRequestParams(
        temperature=LLM_TEMPERATURE,
        max_tokens=FIRST_PASS_LLM_MAX_TOKENS,
        top_p=LLM_TOP_P,
        seed=LLM_SEED,
    )


def write_annotation_outputs(
    output_dir: Path,
    tasks: list[AnnotationTask],
    records: list[LLMAnnotationRecord],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = output_dir / "llm_annotations.jsonl"
    summary_path = output_dir / "summary.json"
    write_jsonl(annotations_path, model_rows(records))
    summary = {
        "total": len(tasks),
        "audit_label_counts": dict(Counter(task.audit_label for task in tasks)),
        "comment_state_counts": dict(Counter(task.context.comment_state for task in tasks)),
        "extend_type_counts": dict(Counter(task.context.extend_type for task in tasks)),
        "llm_annotation_counts": dict(Counter("ok" if item.ok else "failed" for item in records)),
        "outputs": {"llm_annotations": str(annotations_path)},
    }
    write_json(summary_path, summary)
    return summary


def collect_async_annotations(
    client: CmbAsyncLLM,
    run_dir: Path,
    output_dir: Path,
    valid_rubrics: set[str],
) -> dict[str, Any]:
    manifest = load_manifest(run_dir)
    tasks = load_tasks(manifest)
    collected = collect_latest_results(client, run_dir)
    records: list[LLMAnnotationRecord] = []
    failed_ids: list[str] = []
    for task in tasks:
        comment_id = str(task.comment_id)
        if comment_id in collected.errors:
            record = failed_record(task, collected.errors[comment_id])
        else:
            item = collected.items.get(comment_id)
            record = (
                parse_response_record(task, item.response or {}, valid_rubrics)
                if item is not None
                else failed_record(task, "collected result is missing")
            )
        records.append(record)
        if not record.ok:
            failed_ids.append(comment_id)
    mark_failed_comments(run_dir, failed_ids)
    return write_annotation_outputs(output_dir, tasks, records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and annotate Content RM data.")
    parser.add_argument("--input", type=Path, required=True, help="Raw company comment JSONL.")
    parser.add_argument("--rubrics", type=Path, required=True, help="Rubrics markdown file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Generated artifact directory.")
    parser.add_argument("--write-normalized", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--call-llm", action="store_true", help="Run synchronous first-pass annotation.")
    mode.add_argument(
        "--async-action",
        choices=["plan", "submit", "status", "collect", "retry-failed"],
    )
    parser.add_argument("--run-dir", type=Path, help="Async run directory.")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    needs_llm_config = args.call_llm or (
        args.async_action is not None and args.async_action != "plan"
    )
    if needs_llm_config and not (
        LLM_BASE_URL and LLM_MODEL and LLM_AUTHORIZATION
    ):
        parser.error("LLM_BASE_URL, LLM_MODEL, and LLM_AUTHORIZATION are required")
    return args


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    rubrics = read_rubrics(args.rubrics)
    valid_rubrics = {item["name"] for item in rubrics}
    tasks = build_tasks(rows, rubrics)
    run_dir = args.run_dir or args.output_dir / "async" / "first_pass"
    output_path = args.output_dir / "llm_annotations.jsonl"

    if args.write_normalized:
        write_jsonl(
            args.output_dir / "normalized_comments.jsonl",
            [
                {
                    "comment_id": str(task.comment_id),
                    "audit_label": task.audit_label,
                    "context": task.context.model_dump(mode="json"),
                }
                for task in tasks
            ],
        )

    if args.async_action == "plan":
        print(json.dumps(plan_summary(tasks, args.batch_size), ensure_ascii=False, indent=2))
        return

    if args.async_action:
        if args.async_action == "submit":
            initialize_run(
                run_dir=run_dir,
                stage="first_pass",
                tasks=tasks,
                input_path=args.input,
                rubrics_path=args.rubrics,
                output_path=output_path,
                model=LLM_MODEL,
                batch_size=args.batch_size,
                params=request_params(),
            )
        else:
            if not manifest_path(run_dir).exists():
                raise ValueError(f"async run does not exist: {run_dir}")
            initialize_run(
                run_dir=run_dir,
                stage="first_pass",
                tasks=tasks,
                input_path=args.input,
                rubrics_path=args.rubrics,
                output_path=output_path,
                model=LLM_MODEL,
                batch_size=args.batch_size,
                params=request_params(),
            )

        with make_async_client() as client:
            if args.async_action == "submit":
                result = submit_run(client, run_dir)
            elif args.async_action == "status":
                result = status_summary(refresh_status(client, run_dir))
            elif args.async_action == "collect":
                result = collect_async_annotations(client, run_dir, args.output_dir, valid_rubrics)
            else:
                append_retry_jobs(run_dir)
                result = submit_run(client, run_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.call_llm:
        records = run_sync_annotations(tasks, valid_rubrics)
        summary = write_annotation_outputs(args.output_dir, tasks, records)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "total": len(tasks),
            "audit_label_counts": dict(Counter(task.audit_label for task in tasks)),
            "outputs": {},
        }
        write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
