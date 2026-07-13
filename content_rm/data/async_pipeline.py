"""Recoverable batch orchestration shared by first- and second-pass annotation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from cmb_async_model import (
    TERMINAL_STATES,
    AsyncResultItem,
    AsyncTaskContent,
    AsyncTaskParams,
    CmbAsyncLLM,
    TaskState,
)
from common import read_jsonl, write_json, write_jsonl
from records import (
    AnnotationTask,
    AsyncJobRecord,
    AsyncRequestParams,
    AsyncRunManifest,
    RunStage,
    model_rows,
    read_model_jsonl,
)

MANIFEST_NAME = "manifest.json"
TASK_SNAPSHOT_NAME = "tasks.jsonl"


@dataclass
class CollectedResults:
    items: dict[str, AsyncResultItem]
    errors: dict[str, str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(run_dir: Path) -> Path:
    return run_dir / MANIFEST_NAME


def load_manifest(run_dir: Path) -> AsyncRunManifest:
    path = manifest_path(run_dir)
    if not path.exists():
        raise ValueError(f"async run manifest does not exist: {path}")
    return AsyncRunManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))


def save_manifest(run_dir: Path, manifest: AsyncRunManifest) -> None:
    write_json(manifest_path(run_dir), manifest.model_dump(mode="json"))


def load_tasks(manifest: AsyncRunManifest) -> list[AnnotationTask]:
    return read_model_jsonl(Path(manifest.task_snapshot), AnnotationTask)


def _chunks(values: list[UUID], size: int) -> list[list[UUID]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def plan_summary(tasks: list[AnnotationTask], batch_size: int) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    serialized_bytes = sum(
        len(json.dumps(task.model_dump(mode="json"), ensure_ascii=False).encode("utf-8"))
        for task in tasks
    )
    return {
        "tasks": len(tasks),
        "batch_size": batch_size,
        "batches": (len(tasks) + batch_size - 1) // batch_size if tasks else 0,
        "estimated_task_payload_bytes": serialized_bytes,
    }


def initialize_run(
    *,
    run_dir: Path,
    stage: RunStage,
    tasks: list[AnnotationTask],
    input_path: Path,
    rubrics_path: Path,
    output_path: Path,
    model: str,
    batch_size: int,
    params: AsyncRequestParams,
) -> AsyncRunManifest:
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    existing_path = manifest_path(run_dir)
    input_hash = sha256_file(input_path)
    rubrics_hash = sha256_file(rubrics_path)
    task_snapshot = run_dir / TASK_SNAPSHOT_NAME

    if existing_path.exists():
        manifest = load_manifest(run_dir)
        expected = {
            "stage": stage,
            "input_sha256": input_hash,
            "rubrics_sha256": rubrics_hash,
            "output_path": str(output_path.resolve()),
            "model": model,
        }
        actual = {
            "stage": manifest.stage,
            "input_sha256": manifest.input_sha256,
            "rubrics_sha256": manifest.rubrics_sha256,
            "output_path": manifest.output_path,
            "model": manifest.model,
        }
        if actual != expected:
            raise ValueError(
                "existing async run does not match current input/configuration; "
                "choose a different --run-dir"
            )
        saved_tasks = load_tasks(manifest)
        if [task.comment_id for task in saved_tasks] != [task.comment_id for task in tasks]:
            raise ValueError("existing async task snapshot does not match current task order")
        return manifest

    write_jsonl(task_snapshot, model_rows(tasks))
    comment_ids = [task.comment_id for task in tasks]
    jobs = [AsyncJobRecord(comment_ids=chunk) for chunk in _chunks(comment_ids, batch_size)]
    manifest = AsyncRunManifest(
        stage=stage,
        input_path=str(input_path.resolve()),
        input_sha256=input_hash,
        rubrics_path=str(rubrics_path.resolve()),
        rubrics_sha256=rubrics_hash,
        output_path=str(output_path.resolve()),
        model=model,
        batch_size=batch_size,
        params=params,
        task_snapshot=str(task_snapshot),
        jobs=jobs,
    )
    save_manifest(run_dir, manifest)
    return manifest


def _client_params(params: AsyncRequestParams) -> AsyncTaskParams:
    return AsyncTaskParams.model_validate(params.model_dump())


def _contents_for_job(
    manifest: AsyncRunManifest,
    job: AsyncJobRecord,
    tasks_by_id: dict[str, AnnotationTask],
) -> list[AsyncTaskContent]:
    contents: list[AsyncTaskContent] = []
    for comment_id in job.comment_ids:
        task = tasks_by_id[str(comment_id)]
        contents.append(
            AsyncTaskContent(
                custom_id=str(comment_id),
                messages=task.messages,
                model=manifest.model,
            )
        )
    return contents


def _update_job_from_detail(job: AsyncJobRecord, detail: Any) -> None:
    job.platform_state = detail.state.value if detail.state else None
    job.total = detail.total_infer_num
    job.success = detail.success_num
    job.failed = detail.failed_num
    job.error = detail.reason


def submit_run(client: CmbAsyncLLM, run_dir: Path) -> dict[str, Any]:
    manifest = load_manifest(run_dir)
    tasks = load_tasks(manifest)
    tasks_by_id = {str(task.comment_id): task for task in tasks}

    for index, job in enumerate(manifest.jobs, start=1):
        try:
            if job.lifecycle == "planned":
                job.task_id = client.init_batch_task(
                    job_name=f"content-rm-{manifest.stage}-{index}-attempt-{job.attempt}"
                )
                job.lifecycle = "initialized"
                job.error = None
                save_manifest(run_dir, manifest)
                resumed = False
            else:
                resumed = True

            if not job.task_id:
                raise ValueError("initialized job is missing task_id")

            if resumed and job.lifecycle in {"initialized", "uploaded"}:
                detail = client.get_task_detail(job.task_id)
                _update_job_from_detail(job, detail)
                if detail.state not in {TaskState.PREPARE, TaskState.UPLOADING}:
                    job.lifecycle = "submitted"
                elif job.lifecycle == "initialized" and detail.total_infer_num == len(job.comment_ids):
                    job.lifecycle = "uploaded"
                save_manifest(run_dir, manifest)

            if job.lifecycle == "initialized":
                client.upload_batch_content(
                    job.task_id,
                    _contents_for_job(manifest, job, tasks_by_id),
                    _client_params(manifest.params),
                )
                job.lifecycle = "uploaded"
                job.error = None
                save_manifest(run_dir, manifest)

            if job.lifecycle == "uploaded":
                client.submit_batch_task(job.task_id)
                job.lifecycle = "submitted"
                job.error = None
                save_manifest(run_dir, manifest)
        except Exception as exc:
            job.error = str(exc)
            save_manifest(run_dir, manifest)
            raise

    return status_summary(manifest)


def refresh_status(client: CmbAsyncLLM, run_dir: Path) -> AsyncRunManifest:
    manifest = load_manifest(run_dir)
    for job in manifest.jobs:
        if not job.task_id:
            continue
        try:
            detail = client.get_task_detail(job.task_id)
            _update_job_from_detail(job, detail)
        except Exception as exc:
            job.error = str(exc)
    save_manifest(run_dir, manifest)
    return manifest


def status_summary(manifest: AsyncRunManifest) -> dict[str, Any]:
    state_counts = Counter(job.platform_state or job.lifecycle for job in manifest.jobs)
    return {
        "stage": manifest.stage,
        "tasks": sum(len(job.comment_ids) for job in manifest.jobs if job.attempt == 1),
        "jobs": len(manifest.jobs),
        "job_state_counts": dict(state_counts),
        "failed_comment_ids": len(manifest.failed_comment_ids),
    }


def _latest_attempts(manifest: AsyncRunManifest) -> dict[str, int]:
    latest_attempt_by_id: dict[str, int] = {}
    for job in manifest.jobs:
        for comment_id in job.comment_ids:
            key = str(comment_id)
            latest_attempt_by_id[key] = max(latest_attempt_by_id.get(key, 0), job.attempt)
    return latest_attempt_by_id


def _latest_jobs(manifest: AsyncRunManifest) -> list[AsyncJobRecord]:
    latest_attempt_by_id = _latest_attempts(manifest)
    return [
        job
        for job in manifest.jobs
        if any(latest_attempt_by_id[str(comment_id)] == job.attempt for comment_id in job.comment_ids)
    ]


def collect_latest_results(client: CmbAsyncLLM, run_dir: Path) -> CollectedResults:
    manifest = refresh_status(client, run_dir)
    latest_attempt_by_id = _latest_attempts(manifest)
    latest_jobs = _latest_jobs(manifest)
    active = [
        job.task_id or "uninitialized"
        for job in latest_jobs
        if job.lifecycle != "submitted"
        or job.platform_state is None
        or TaskState(job.platform_state) not in TERMINAL_STATES
    ]
    if active:
        raise ValueError(f"async jobs are not terminal yet: {active}")

    expected_ids = {str(comment_id) for job in latest_jobs for comment_id in job.comment_ids}
    items: dict[str, AsyncResultItem] = {}
    errors: dict[str, str] = {}
    raw_dir = run_dir / "raw_results"

    for job in latest_jobs:
        job_ids = {
            str(comment_id)
            for comment_id in job.comment_ids
            if latest_attempt_by_id[str(comment_id)] == job.attempt
        }
        state = TaskState(job.platform_state) if job.platform_state else None
        if state != TaskState.COMPLETED:
            reason = job.error or f"platform task ended in state {job.platform_state}"
            errors.update({comment_id: reason for comment_id in job_ids})
            continue

        if not job.task_id:
            errors.update({comment_id: "completed job is missing task_id" for comment_id in job_ids})
            continue
        raw_path = raw_dir / f"{job.task_id}.jsonl"
        if raw_path.exists():
            raw_items = [AsyncResultItem.model_validate(row) for row in read_jsonl(raw_path)]
        else:
            raw_items = client.get_all_results(job.task_id)
            write_jsonl(raw_path, [item.model_dump(mode="json") for item in raw_items])

        seen_in_job: set[str] = set()
        for item in raw_items:
            custom_id = item.custom_id or ""
            if custom_id not in expected_ids:
                raise ValueError(f"platform returned unexpected custom_id: {custom_id!r}")
            if custom_id not in job_ids:
                # This item belongs to an older attempt in a mixed batch. Its
                # latest retry result is authoritative, so ignore the stale one.
                continue
            if custom_id in seen_in_job:
                errors[custom_id] = "platform returned duplicate results for custom_id"
                items.pop(custom_id, None)
                continue
            seen_in_job.add(custom_id)
            if item.error:
                errors[custom_id] = str(item.error)
            elif item.response is None:
                errors[custom_id] = "platform result is missing response"
            else:
                items[custom_id] = item
        for missing_id in job_ids - seen_in_job:
            errors[missing_id] = "platform did not return a result"

    return CollectedResults(items=items, errors=errors)


def mark_failed_comments(run_dir: Path, failed_comment_ids: list[str]) -> AsyncRunManifest:
    manifest = load_manifest(run_dir)
    task_order = [str(task.comment_id) for task in load_tasks(manifest)]
    failed = set(failed_comment_ids)
    manifest.failed_comment_ids = [UUID(comment_id) for comment_id in task_order if comment_id in failed]
    save_manifest(run_dir, manifest)
    return manifest


def append_retry_jobs(run_dir: Path) -> AsyncRunManifest:
    manifest = load_manifest(run_dir)
    if not manifest.failed_comment_ids:
        raise ValueError("manifest has no failed comments to retry; run collect first")
    next_attempt = max((job.attempt for job in manifest.jobs), default=0) + 1
    for chunk in _chunks(manifest.failed_comment_ids, manifest.batch_size):
        manifest.jobs.append(AsyncJobRecord(comment_ids=chunk, attempt=next_attempt))
    manifest.failed_comment_ids = []
    save_manifest(run_dir, manifest)
    return manifest
