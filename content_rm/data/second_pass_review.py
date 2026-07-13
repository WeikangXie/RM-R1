#!/usr/bin/env python3
"""Run label-conditioned second-pass review for first-pass disagreements."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests

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
    read_rubrics,
    rubrics_text,
    write_json,
    write_jsonl,
)
from config import (
    LLM_AUTHORIZATION,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_RETRIES,
    LLM_RETRY_SLEEP,
    LLM_SEED,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    LLM_TOP_P,
    SECOND_PASS_LLM_MAX_TOKENS,
)
from records import (
    AnnotationTask,
    AsyncRequestParams,
    LLMAnnotationRecord,
    Message,
    SecondPassAnnotation,
    SecondPassRecord,
    model_rows,
    read_model_jsonl,
)

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
3. 如果无法合理解释，status 输出 need_review，不要强行编造理由。
4. decision=pass 时 violated_rubrics 通常应为空数组。
5. decision=reject 时 violated_rubrics 应尽量给出命中的规则。
"""


def user_prompt(record: LLMAnnotationRecord, rubrics: list[dict[str, str]]) -> str:
    if record.annotation is None:
        raise ValueError("second-pass prompt requires a successful first-pass annotation")
    context = record.context
    annotation = record.annotation
    return f"""请对下面这条金融内容社区 AI 回复做二次复核推理。

Rubrics:
{rubrics_text(rubrics)}

运营最终审核结果：
{record.audit_label}

第一次 LLM 审核结果：
- decision: {annotation.decision}
- violated_rubrics: {" | ".join(annotation.violated_rubrics)}
- reasoning: {annotation.reasoning}

业务上下文：
- 原帖/父评论：{context.text}
- 产品名称：{context.product_name}
- 话题：{" | ".join(context.topic_titles)}
- 圈子：{" | ".join(context.coterie_names)}
- 回复类型：{context.extend_type}

待审核 AI 回复：
{context.ai_reply}

请判断运营最终审核结果是否能被 rubrics 合理解释。能解释则 status=ok；不能解释则 status=need_review。
"""


def load_first_pass(path: Path) -> list[LLMAnnotationRecord]:
    records = read_model_jsonl(path, LLMAnnotationRecord)
    seen: set[str] = set()
    for record in records:
        comment_id = str(record.comment_id)
        if comment_id in seen:
            raise ValueError(f"duplicate comment_id in llm_annotations: {comment_id}")
        seen.add(comment_id)
    return records


def build_tasks(
    records: list[LLMAnnotationRecord], rubrics: list[dict[str, str]]
) -> list[AnnotationTask]:
    tasks: list[AnnotationTask] = []
    for record in records:
        if not record.ok or record.annotation is None:
            continue
        if record.annotation.decision == record.audit_label:
            continue
        tasks.append(
            AnnotationTask(
                comment_id=record.comment_id,
                audit_label=record.audit_label,
                context=record.context,
                first_annotation=record.annotation,
                messages=[
                    Message(role="system", content=SYSTEM_PROMPT),
                    Message(role="user", content=user_prompt(record, rubrics)),
                ],
            )
        )
    return tasks


def validate_annotation(
    value: dict[str, Any], expected_decision: str, valid_rubrics: set[str]
) -> SecondPassAnnotation:
    annotation = SecondPassAnnotation.model_validate(value)
    if annotation.decision != expected_decision:
        raise ValueError(
            f"decision must equal audit_label {expected_decision!r}, got {annotation.decision!r}"
        )
    unknown = [item for item in annotation.violated_rubrics if item not in valid_rubrics]
    if unknown:
        raise ValueError(f"unknown rubrics: {unknown}")
    return annotation


def failed_record(task: AnnotationTask, error: str, raw_content: str | None = None) -> SecondPassRecord:
    return SecondPassRecord(
        comment_id=task.comment_id,
        ok=False,
        raw_content=raw_content,
        error=error,
    )


def parse_response_record(
    task: AnnotationTask,
    response: dict[str, Any],
    valid_rubrics: set[str],
) -> SecondPassRecord:
    raw_content: str | None = None
    try:
        raw_content = extract_completion_content(response)
        annotation = validate_annotation(
            extract_json_object(raw_content), task.audit_label, valid_rubrics
        )
        return SecondPassRecord(
            comment_id=task.comment_id,
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
        "max_tokens": SECOND_PASS_LLM_MAX_TOKENS,
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
) -> SecondPassRecord:
    last_error: Exception | None = None
    last_failed_record: SecondPassRecord | None = None
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
) -> list[SecondPassRecord]:
    session = requests.Session()
    session.headers.update(
        {"Authorization": LLM_AUTHORIZATION, "Content-Type": "application/json"}
    )
    try:
        results: list[SecondPassRecord] = []
        for index, task in enumerate(tasks, start=1):
            print(f"Second-pass annotating {index}/{len(tasks)}: {task.comment_id}")
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
        max_tokens=SECOND_PASS_LLM_MAX_TOKENS,
        top_p=LLM_TOP_P,
        seed=LLM_SEED,
    )


def make_summary(
    first_pass: list[LLMAnnotationRecord],
    tasks: list[AnnotationTask],
    results: list[SecondPassRecord] | None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total_first_pass_rows": len(first_pass),
        "selected_disagreements": len(tasks),
        "audit_label_counts": dict(Counter(record.audit_label for record in first_pass)),
        "first_pass_decision_counts": dict(
            Counter(
                record.annotation.decision
                for record in first_pass
                if record.ok and record.annotation is not None
            )
        ),
        "disagreement_label_pairs": dict(
            Counter(f"{task.first_annotation.decision}->{task.audit_label}" for task in tasks if task.first_annotation)
        ),
    }
    if results is not None:
        summary["second_pass_counts"] = dict(
            Counter("ok" if result.ok else "failed" for result in results)
        )
        summary["second_pass_status_counts"] = dict(
            Counter(
                result.annotation.status
                for result in results
                if result.ok and result.annotation is not None
            )
        )
    return summary


def write_outputs(
    output: Path,
    first_pass: list[LLMAnnotationRecord],
    tasks: list[AnnotationTask],
    records: list[SecondPassRecord],
) -> dict[str, Any]:
    write_jsonl(output, model_rows(records))
    summary_path = output.parent / "second_pass_summary.json"
    summary = make_summary(first_pass, tasks, records)
    summary["outputs"] = {
        "second_pass_annotations": str(output),
        "summary": str(summary_path),
    }
    write_json(summary_path, summary)
    return summary


def collect_async_annotations(
    client: CmbAsyncLLM,
    run_dir: Path,
    output: Path,
    first_pass: list[LLMAnnotationRecord],
    valid_rubrics: set[str],
) -> dict[str, Any]:
    manifest = load_manifest(run_dir)
    tasks = load_tasks(manifest)
    collected = collect_latest_results(client, run_dir)
    records: list[SecondPassRecord] = []
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
    return write_outputs(output, first_pass, tasks, records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run label-conditioned second-pass review.")
    parser.add_argument("--llm-annotations", type=Path, required=True)
    parser.add_argument("--rubrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--async-action",
        choices=["plan", "submit", "status", "collect", "retry-failed"],
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    if args.dry_run and args.async_action:
        parser.error("--dry-run and --async-action are mutually exclusive")
    needs_llm_config = not args.dry_run and args.async_action != "plan"
    if needs_llm_config and not (LLM_BASE_URL and LLM_MODEL and LLM_AUTHORIZATION):
        parser.error("LLM_BASE_URL, LLM_MODEL, and LLM_AUTHORIZATION are required")
    return args


def main() -> None:
    args = parse_args()
    first_pass = load_first_pass(args.llm_annotations)
    rubrics = read_rubrics(args.rubrics)
    valid_rubrics = {item["name"] for item in rubrics}
    tasks = build_tasks(first_pass, rubrics)
    run_dir = args.run_dir or args.output.parent / "async" / "second_pass"

    if args.dry_run:
        print(json.dumps(make_summary(first_pass, tasks, None), ensure_ascii=False, indent=2))
        return
    if args.async_action == "plan":
        print(json.dumps(plan_summary(tasks, args.batch_size), ensure_ascii=False, indent=2))
        return

    if args.async_action:
        if args.async_action == "submit":
            initialize_run(
                run_dir=run_dir,
                stage="second_pass",
                tasks=tasks,
                input_path=args.llm_annotations,
                rubrics_path=args.rubrics,
                output_path=args.output,
                model=LLM_MODEL,
                batch_size=args.batch_size,
                params=request_params(),
            )
        else:
            if not manifest_path(run_dir).exists():
                raise ValueError(f"async run does not exist: {run_dir}")
            initialize_run(
                run_dir=run_dir,
                stage="second_pass",
                tasks=tasks,
                input_path=args.llm_annotations,
                rubrics_path=args.rubrics,
                output_path=args.output,
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
                result = collect_async_annotations(
                    client, run_dir, args.output, first_pass, valid_rubrics
                )
            else:
                append_retry_jobs(run_dir)
                result = submit_run(client, run_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    records = run_sync_annotations(tasks, valid_rubrics)
    summary = write_outputs(args.output, first_pass, tasks, records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
