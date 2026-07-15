#!/usr/bin/env python3
"""Run label-conditioned second-pass review for first-pass disagreements."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

CONTENT_RM_ROOT = Path(__file__).resolve().parents[1]
if str(CONTENT_RM_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTENT_RM_ROOT))

import requests

from infrastructure.async_pipeline import (
    AsyncBatchTask,
    AsyncRequestParams,
    append_retry_jobs,
    collect_latest_results,
    initialize_run,
    manifest_path,
    mark_failed_tasks,
    plan_summary,
    refresh_status,
    status_summary,
    submit_run,
)
from infrastructure.cmb_async_model import CmbAsyncLLM
from infrastructure.common import (
    build_llm_url,
    extract_completion_content,
    extract_json_object,
    model_rows,
    read_model_jsonl,
    write_json,
    write_jsonl,
)
from infrastructure.config import (
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
from data.annotation_config import SECOND_PASS_LLM_MAX_TOKENS
from data.records import (
    AnnotationTask,
    LLMAnnotationRecord,
    Message,
    SecondPassAnnotation,
    SecondPassRecord,
    decision_rubrics_are_consistent,
)
from data.rubric_utils import canonicalize_rubric_names, read_rubrics, rubrics_text

SYSTEM_PROMPT = """你是金融内容社区的 AI 回复审核员。你的输出将直接作为一条独立审核结论使用。

你会收到一个待验证的候选 decision。请严格依据待审核回复中的具体内容和给定 rubrics 独立审核。候选 decision 只用于比较，不是必须服从的答案。

输出必须是 JSON，不要输出 JSON 之外的文字。JSON 字段固定为：
{
  "status": "ok 或 need_review",
  "violated_rubrics": ["命中的规则名称；如果没有则为空数组"],
  "reasoning": "可独立使用的审核判断，必须引用或明确指出待审核回复中的具体内容",
  "decision": "你依据回复内容独立得出的 pass 或 reject"
}

要求：
1. decision 必须反映你的真实审核判断，不得为了等于候选 decision 而改写。
2. 你的 decision 等于候选 decision，且有明确、具体的文本证据时，status=ok；decision 不等于候选 decision，或者证据不足、需要猜测时，status=need_review。
3. decision=reject 时，必须指出具体风险表述，并给出至少一个规范的 rubric 名称。
4. decision=pass 时，应说明回复实际表达了什么以及为何未触发风险，violated_rubrics 必须为空数组。
5. reasoning、violated_rubrics 和 decision 必须表达同一个审核结论。
6. reasoning 必须像直接审核该回复得出的结论，不得提及“运营”“候选结论”“给定标签”“第一次标注”“二次复核”“rubrics”或“该结果合理”等审核流程信息。
7. 不要使用“根据规则和上下文可以判断”之类的空泛开头。

审核交易引导时，不要只检查“上车、抄底、加仓”等词。明确建议用户购买、赎回、转换、更换产品或份额，例如“优先考虑”“强烈推荐”“建议尽快切换”，也应按交易引导风险判断，不能因为回复同时包含费用或产品分析就忽略。

reasoning 示例：
- 合格 reject：回复使用“后续肯定会上涨”描述未来表现，将不确定收益表述为确定结果，构成收益暗示，因此不通过。
- 合格 reject：回复使用“建议尽快切换至另一份额”直接引导用户转换基金份额，构成交易诱导，因此不通过。
- 合格 pass：回复仅说明历史表现并提示市场风险，没有承诺未来收益、诱导交易或提供个性化投资建议，可以通过。
- 合格 need_review：候选 decision 为 reject，但独立审核结论为 pass 时，输出 status=need_review、decision=pass、空 violated_rubrics，并用 reasoning 直接说明可以通过的内容依据。
- 不合格：根据 rubrics 和上下文，可以判断运营的 reject 是合理的。
"""


def user_prompt(record: LLMAnnotationRecord, rubrics: list[dict[str, str]]) -> str:
    context = record.context
    return f"""请审核下面这条金融内容社区 AI 回复，并验证候选 decision 是否有充分的文本依据。

Rubrics:
{rubrics_text(rubrics)}

待验证的候选 decision：
{record.audit_label}

业务上下文：
- 原帖/父评论：{context.text}
- 产品名称：{context.product_name}
- 话题：{" | ".join(context.topic_titles)}
- 圈子：{" | ".join(context.coterie_names)}
- 回复类型：{context.extend_type}

待审核 AI 回复：
{context.ai_reply}

请先独立得出真实 decision，再与候选 decision 比较。两者一致且证据充分时输出 status=ok；两者不一致或证据不足时输出 status=need_review。不要为了匹配候选 decision 改写判断。
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
        # The first-pass result is only used to select disagreements. Excluding
        # it from messages avoids anchoring and process-style SFT reasoning.
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
    value: dict[str, Any], audit_label: str, valid_rubrics: set[str]
) -> SecondPassAnnotation:
    annotation = SecondPassAnnotation.model_validate(value)
    canonical_rubrics = canonicalize_rubric_names(
        annotation.violated_rubrics, valid_rubrics
    )
    status = annotation.status
    # A label disagreement or inconsistent supervision is a business review
    # outcome, not a transport/parser failure, so retain it as need_review.
    if annotation.decision != audit_label or not decision_rubrics_are_consistent(
        annotation.decision, canonical_rubrics
    ):
        status = "need_review"
    return annotation.model_copy(
        update={
            "status": status,
            "violated_rubrics": canonical_rubrics,
        }
    )


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


def to_async_tasks(tasks: list[AnnotationTask]) -> list[AsyncBatchTask]:
    """Adapt business tasks to the platform-only async contract."""

    return [
        AsyncBatchTask(custom_id=str(task.comment_id), messages=task.messages)
        for task in tasks
    ]


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
    tasks: list[AnnotationTask],
    valid_rubrics: set[str],
) -> dict[str, Any]:
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
    mark_failed_tasks(run_dir, failed_ids)
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
    async_tasks = to_async_tasks(tasks)
    run_dir = args.run_dir or args.output.parent / "async" / "second_pass"

    if args.dry_run:
        print(json.dumps(make_summary(first_pass, tasks, None), ensure_ascii=False, indent=2))
        return
    if args.async_action == "plan":
        print(
            json.dumps(
                plan_summary(async_tasks, args.batch_size),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.async_action:
        if args.async_action != "submit" and not manifest_path(run_dir).exists():
            raise ValueError(f"async run does not exist: {run_dir}")
        async_params = request_params()
        initialize_run(
            run_dir=run_dir,
            run_name="content-rm-second-pass",
            tasks=async_tasks,
            input_paths={
                "llm_annotations": args.llm_annotations,
                "rubrics": args.rubrics,
            },
            output_path=args.output,
            model=LLM_MODEL,
            batch_size=args.batch_size,
            params=async_params,
        )
        with make_async_client() as client:
            if args.async_action == "submit":
                result = submit_run(client, run_dir)
            elif args.async_action == "status":
                result = status_summary(refresh_status(client, run_dir))
            elif args.async_action == "collect":
                result = collect_async_annotations(
                    client,
                    run_dir,
                    args.output,
                    first_pass,
                    tasks,
                    valid_rubrics,
                )
            else:
                append_retry_jobs(run_dir, params=async_params)
                result = submit_run(client, run_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    records = run_sync_annotations(tasks, valid_rubrics)
    summary = write_outputs(args.output, first_pass, tasks, records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
