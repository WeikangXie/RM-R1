"""Recoverable, business-agnostic orchestration for asynchronous LLM batches."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator

from .chat_completions_req import Message, StrictModel
from .cmb_async_model import (
    TERMINAL_STATES,
    AsyncResultItem,
    AsyncTaskContent,
    AsyncTaskParams,
    CmbAsyncLLM,
    TaskState,
)
from .common import model_rows, read_jsonl, read_model_jsonl, write_json, write_jsonl

MANIFEST_NAME = "manifest.json"
TASK_SNAPSHOT_NAME = "tasks.jsonl"

JobLifecycle = Literal["planned", "initialized", "uploaded", "submitted"]


class AsyncBatchTask(StrictModel):
    """One platform request, identified only by its API-level custom ID."""

    custom_id: str = Field(min_length=1, max_length=120)
    messages: list[Message] = Field(default_factory=list)


class AsyncRequestParams(StrictModel):
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    stop: list[str] | None = None


class InputFingerprint(StrictModel):
    path: str
    sha256: str


class AsyncJobRecord(StrictModel):
    custom_ids: list[str]
    attempt: int = 1
    lifecycle: JobLifecycle = "planned"
    task_id: str | None = None
    platform_state: str | None = None
    total: int | None = None
    success: int | None = None
    failed: int | None = None
    error: str | None = None


class AsyncRunManifest(StrictModel):
    schema_version: int = 2
    run_name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    input_files: dict[str, InputFingerprint]
    output_path: str
    model: str
    batch_size: int
    params: AsyncRequestParams
    task_snapshot: str
    jobs: list[AsyncJobRecord] = Field(default_factory=list)
    failed_custom_ids: list[str] = Field(default_factory=list)

    @field_validator("batch_size")
    @classmethod
    def positive_batch_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("batch_size must be positive")
        return value


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


def load_tasks(manifest: AsyncRunManifest) -> list[AsyncBatchTask]:
    return read_model_jsonl(Path(manifest.task_snapshot), AsyncBatchTask)


def _chunks(values: list[str], size: int) -> list[list[str]]:
    if size <= 0:
        raise ValueError("batch_size must be positive")
    return [values[index : index + size] for index in range(0, len(values), size)]


def _validate_unique_custom_ids(tasks: list[AsyncBatchTask]) -> None:
    seen: set[str] = set()
    for task in tasks:
        if task.custom_id in seen:
            raise ValueError(f"duplicate custom_id in async tasks: {task.custom_id}")
        seen.add(task.custom_id)


def plan_summary(tasks: list[AsyncBatchTask], batch_size: int) -> dict[str, Any]:
    _validate_unique_custom_ids(tasks)
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


def _fingerprint_inputs(input_paths: dict[str, Path]) -> dict[str, InputFingerprint]:
    if not input_paths:
        raise ValueError("at least one input file is required")
    return {
        name: InputFingerprint(path=str(path.resolve()), sha256=sha256_file(path))
        for name, path in sorted(input_paths.items())
    }


def initialize_run(
    *,
    run_dir: Path,
    run_name: str,
    tasks: list[AsyncBatchTask],
    input_paths: dict[str, Path],
    output_path: Path,
    model: str,
    batch_size: int,
    params: AsyncRequestParams,
) -> AsyncRunManifest:
    """Create a run or verify that an existing run matches the same inputs."""

    if not run_name.strip():
        raise ValueError("run_name must not be empty")
    if not model:
        raise ValueError("model must not be empty")
    _validate_unique_custom_ids(tasks)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    existing_path = manifest_path(run_dir)
    input_files = _fingerprint_inputs(input_paths)
    task_snapshot = run_dir / TASK_SNAPSHOT_NAME

    if existing_path.exists():
        manifest = load_manifest(run_dir)
        expected = {
            "run_name": run_name,
            "input_files": {
                name: value.model_dump(mode="json") for name, value in input_files.items()
            },
            "output_path": str(output_path.resolve()),
            "model": model,
            "batch_size": batch_size,
            "params": params.model_dump(mode="json"),
        }
        actual = {
            "run_name": manifest.run_name,
            "input_files": {
                name: value.model_dump(mode="json")
                for name, value in manifest.input_files.items()
            },
            "output_path": manifest.output_path,
            "model": manifest.model,
            "batch_size": manifest.batch_size,
            "params": manifest.params.model_dump(mode="json"),
        }
        if actual != expected:
            raise ValueError(
                "existing async run does not match current input/configuration; "
                "choose a different --run-dir"
            )
        saved_tasks = load_tasks(manifest)
        if model_rows(saved_tasks) != model_rows(tasks):
            raise ValueError("existing async task snapshot does not match current tasks")
        return manifest

    write_jsonl(task_snapshot, model_rows(tasks))
    custom_ids = [task.custom_id for task in tasks]
    jobs = [AsyncJobRecord(custom_ids=chunk) for chunk in _chunks(custom_ids, batch_size)]
    manifest = AsyncRunManifest(
        run_name=run_name,
        input_files=input_files,
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
    tasks_by_id: dict[str, AsyncBatchTask],
) -> list[AsyncTaskContent]:
    return [
        AsyncTaskContent(
            custom_id=custom_id,
            messages=tasks_by_id[custom_id].messages,
            model=manifest.model,
        )
        for custom_id in job.custom_ids
    ]


def _update_job_from_detail(job: AsyncJobRecord, detail: Any) -> None:
    job.platform_state = detail.state.value if detail.state else None
    job.total = detail.total_infer_num
    job.success = detail.success_num
    job.failed = detail.failed_num
    job.error = detail.reason


def submit_run(client: CmbAsyncLLM, run_dir: Path) -> dict[str, Any]:
    manifest = load_manifest(run_dir)
    tasks = load_tasks(manifest)
    tasks_by_id = {task.custom_id: task for task in tasks}

    for index, job in enumerate(manifest.jobs, start=1):
        try:
            if job.lifecycle == "planned":
                job.task_id = client.init_batch_task(
                    job_name=f"{manifest.run_name}-{index}-attempt-{job.attempt}"
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
                elif (
                    job.lifecycle == "initialized"
                    and detail.total_infer_num == len(job.custom_ids)
                ):
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
        "run_name": manifest.run_name,
        "tasks": sum(len(job.custom_ids) for job in manifest.jobs if job.attempt == 1),
        "jobs": len(manifest.jobs),
        "job_state_counts": dict(state_counts),
        "failed_custom_ids": len(manifest.failed_custom_ids),
    }


def _latest_attempts(manifest: AsyncRunManifest) -> dict[str, int]:
    latest_attempt_by_id: dict[str, int] = {}
    for job in manifest.jobs:
        for custom_id in job.custom_ids:
            latest_attempt_by_id[custom_id] = max(
                latest_attempt_by_id.get(custom_id, 0), job.attempt
            )
    return latest_attempt_by_id


def _latest_jobs(manifest: AsyncRunManifest) -> list[AsyncJobRecord]:
    latest_attempt_by_id = _latest_attempts(manifest)
    return [
        job
        for job in manifest.jobs
        if any(
            latest_attempt_by_id[custom_id] == job.attempt
            for custom_id in job.custom_ids
        )
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

    expected_ids = {custom_id for job in latest_jobs for custom_id in job.custom_ids}
    items: dict[str, AsyncResultItem] = {}
    errors: dict[str, str] = {}
    raw_dir = run_dir / "raw_results"

    for job in latest_jobs:
        job_ids = {
            custom_id
            for custom_id in job.custom_ids
            if latest_attempt_by_id[custom_id] == job.attempt
        }
        state = TaskState(job.platform_state) if job.platform_state else None
        if state != TaskState.COMPLETED:
            reason = job.error or f"platform task ended in state {job.platform_state}"
            errors.update({custom_id: reason for custom_id in job_ids})
            continue

        if not job.task_id:
            errors.update(
                {custom_id: "completed job is missing task_id" for custom_id in job_ids}
            )
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
                # A newer retry is authoritative for this item.
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


def mark_failed_tasks(run_dir: Path, failed_custom_ids: list[str]) -> AsyncRunManifest:
    manifest = load_manifest(run_dir)
    task_order = [task.custom_id for task in load_tasks(manifest)]
    failed = set(failed_custom_ids)
    unknown = failed - set(task_order)
    if unknown:
        raise ValueError(f"failed custom_id values are not in this run: {sorted(unknown)}")
    manifest.failed_custom_ids = [custom_id for custom_id in task_order if custom_id in failed]
    save_manifest(run_dir, manifest)
    return manifest


def append_retry_jobs(run_dir: Path) -> AsyncRunManifest:
    manifest = load_manifest(run_dir)
    if not manifest.failed_custom_ids:
        raise ValueError("manifest has no failed tasks to retry; run collect first")
    next_attempt = max((job.attempt for job in manifest.jobs), default=0) + 1
    for chunk in _chunks(manifest.failed_custom_ids, manifest.batch_size):
        manifest.jobs.append(AsyncJobRecord(custom_ids=chunk, attempt=next_attempt))
    manifest.failed_custom_ids = []
    save_manifest(run_dir, manifest)
    return manifest
