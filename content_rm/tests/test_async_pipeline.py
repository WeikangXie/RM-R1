from __future__ import annotations

from pathlib import Path
from uuid import UUID

from infrastructure.async_pipeline import (
    AsyncBatchTask,
    AsyncRequestParams,
    append_retry_jobs,
    collect_latest_results,
    initialize_run,
    load_manifest,
    mark_failed_tasks,
    save_manifest,
    submit_run,
)
from infrastructure.chat_completions_req import Message
from infrastructure.cmb_async_model import (
    AsyncResultItem,
    AsyncTaskContent,
    AsyncTaskDetail,
    TaskState,
)


class FakeAsyncClient:
    def __init__(self) -> None:
        self.counter = 0
        self.details: dict[str, AsyncTaskDetail] = {}
        self.contents: dict[str, list] = {}
        self.omit_once: set[str] = set()
        self.upload_calls = 0
        self.submit_calls = 0

    def init_batch_task(self, job_name: str | None = None) -> str:
        self.counter += 1
        task_id = f"task-{self.counter}"
        self.details[task_id] = AsyncTaskDetail(id=task_id, state=TaskState.PREPARE)
        return task_id

    def upload_batch_content(self, task_id: str, contents: list, params: object) -> None:
        self.upload_calls += 1
        self.contents[task_id] = contents
        self.details[task_id] = AsyncTaskDetail(
            id=task_id,
            state=TaskState.PREPARE,
            totalInferNum=len(contents),
        )

    def submit_batch_task(self, task_id: str) -> None:
        self.submit_calls += 1
        count = len(self.contents[task_id])
        self.details[task_id] = AsyncTaskDetail(
            id=task_id,
            state=TaskState.COMPLETED,
            totalInferNum=count,
            successNum=count,
            failedNum=0,
        )

    def get_task_detail(self, task_id: str) -> AsyncTaskDetail:
        return self.details[task_id]

    def get_all_results(self, task_id: str) -> list[AsyncResultItem]:
        results = []
        for content in self.contents[task_id]:
            if content.custom_id in self.omit_once:
                self.omit_once.remove(content.custom_id)
                continue
            results.append(
                AsyncResultItem(
                    custom_id=content.custom_id,
                    response={
                        "choices": [
                            {
                                "message": {
                                    "content": '{"violated_rubrics": [], "reasoning": "ok", "decision": "pass"}'
                                }
                            }
                        ]
                    },
                )
            )
        return list(reversed(results))


def make_tasks(count: int) -> list[AsyncBatchTask]:
    return [
        AsyncBatchTask(
            custom_id=str(UUID(int=index + 1)),
            messages=[Message(role="user", content="prompt")],
        )
        for index in range(count)
    ]


def create_run(
    tmp_path: Path, count: int, batch_size: int = 500
) -> tuple[Path, list[AsyncBatchTask]]:
    source = tmp_path / "input.jsonl"
    rubrics = tmp_path / "rubrics.md"
    output = tmp_path / "llm_annotations.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    rubrics.write_text("| Rubric | desc |\n| --- | --- |\n| x | y |\n", encoding="utf-8")
    tasks = make_tasks(count)
    run_dir = tmp_path / "run"
    initialize_run(
        run_dir=run_dir,
        run_name="test-first-pass",
        tasks=tasks,
        input_paths={"input": source, "rubrics": rubrics},
        output_path=output,
        model="model",
        batch_size=batch_size,
        params=AsyncRequestParams(temperature=0.0, max_tokens=10),
    )
    return run_dir, tasks


def test_batches_and_custom_ids_are_comment_ids(tmp_path: Path) -> None:
    run_dir, tasks = create_run(tmp_path, 1001)
    manifest = load_manifest(run_dir)
    assert [len(job.custom_ids) for job in manifest.jobs] == [500, 500, 1]
    assert "authorization" not in (run_dir / "manifest.json").read_text(encoding="utf-8").lower()

    client = FakeAsyncClient()
    submit_run(client, run_dir)
    uploaded = [content for batch in client.contents.values() for content in batch]
    assert [content.custom_id for content in uploaded] == [task.custom_id for task in tasks]

    collected = collect_latest_results(client, run_dir)
    assert not collected.errors
    assert set(collected.items) == {task.custom_id for task in tasks}


def test_failed_item_can_be_retried_without_resubmitting_successes(tmp_path: Path) -> None:
    run_dir, tasks = create_run(tmp_path, 3, batch_size=2)
    failed_id = tasks[1].custom_id
    client = FakeAsyncClient()
    client.omit_once.add(failed_id)
    submit_run(client, run_dir)
    first = collect_latest_results(client, run_dir)
    assert first.errors == {failed_id: "platform did not return a result"}

    mark_failed_tasks(run_dir, list(first.errors))
    append_retry_jobs(run_dir)
    submit_run(client, run_dir)
    second = collect_latest_results(client, run_dir)
    assert not second.errors
    assert set(second.items) == {task.custom_id for task in tasks}
    assert len(client.contents["task-3"]) == 1
    assert client.contents["task-3"][0].custom_id == failed_id


def test_submit_resumes_after_upload_response_was_not_persisted(tmp_path: Path) -> None:
    run_dir, tasks = create_run(tmp_path, 2)
    client = FakeAsyncClient()
    task_id = client.init_batch_task()
    client.contents[task_id] = []
    for task in tasks:
        client.contents[task_id].append(
            AsyncTaskContent(
                custom_id=task.custom_id,
                messages=task.messages,
                model="model",
            )
        )
    client.details[task_id] = AsyncTaskDetail(
        id=task_id,
        state=TaskState.PREPARE,
        totalInferNum=2,
    )
    manifest = load_manifest(run_dir)
    manifest.jobs[0].task_id = task_id
    manifest.jobs[0].lifecycle = "initialized"
    save_manifest(run_dir, manifest)

    submit_run(client, run_dir)
    assert client.upload_calls == 0
    assert client.submit_calls == 1
    assert load_manifest(run_dir).jobs[0].lifecycle == "submitted"
