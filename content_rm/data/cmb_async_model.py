"""Client for the company platform's asynchronous chat-completions API."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any
from urllib.parse import urljoin

import requests
from pydantic import ConfigDict, Field

from records import Message, StrictModel


class AsyncLLMError(RuntimeError):
    """Raised when the asynchronous platform rejects or cannot serve a request."""


class TaskState(str, Enum):
    PREPARE = "PREPARE"
    CANCEL = "CANCEL"
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    STOP = "STOP"
    INTERRUPT = "INTERRUPT"
    UPLOADING = "UPLOADING"


TERMINAL_FAILURE_STATES = {
    TaskState.CANCEL,
    TaskState.FAILED,
    TaskState.INTERRUPT,
    TaskState.STOP,
}
TERMINAL_STATES = TERMINAL_FAILURE_STATES | {TaskState.COMPLETED}


class CallbackConfig(StrictModel):
    callback_rules: list[str] = Field(default_factory=lambda: ["COMPLETED", "FAILED"])
    callback_type: str = "HTTP"
    url: str
    method: str = "POST"
    headers: dict[str, str] = Field(default_factory=dict)


class AsyncTaskContent(StrictModel):
    custom_id: str = Field(max_length=120)
    messages: list[Message] = Field(default_factory=list)
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    stop: list[str] | None = None


class AsyncTaskParams(StrictModel):
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    stop: list[str] | None = None


class AsyncTaskDetail(StrictModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    job_name: str | None = Field(default=None, alias="jobName")
    model_name: str | None = Field(default=None, alias="modelName")
    total_infer_num: int | None = Field(default=None, alias="totalInferNum")
    processed_index: int | None = Field(default=None, alias="processedIndex")
    state: TaskState | None = None
    reason: str | None = None
    success_num: int | None = Field(default=None, alias="successNum")
    failed_num: int | None = Field(default=None, alias="failedNum")
    prompt_tokens: int | None = Field(default=None, alias="promptTokens")
    completion_tokens: int | None = Field(default=None, alias="completionTokens")
    cost_time: int | None = Field(default=None, alias="costTime")


class AsyncResultItem(StrictModel):
    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    custom_id: str | None = None
    response: dict[str, Any] | None = None
    error: Any = None


class AsyncResultPage(StrictModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    current_page: int = Field(default=1, alias="currentPage")
    page_size: int = Field(default=200, alias="pageSize")
    total: int = 0
    total_pages: int = Field(default=0, alias="totalPages")
    data: list[AsyncResultItem] = Field(default_factory=list)


class CmbAsyncLLM:
    """Small, reusable client for single and batch asynchronous inference."""

    def __init__(
        self,
        host: str,
        authorization: str,
        model: str,
        timeout: float = 60,
        retries: int = 2,
        retry_sleep: float = 1,
        session: requests.Session | None = None,
    ) -> None:
        if not host or not model or not authorization:
            raise ValueError("host, model, and authorization are required")
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retries = max(1, retries)
        self.retry_sleep = retry_sleep
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Authorization": authorization,
                "Content-Type": "application/json",
            }
        )
        self.api_paths = {
            "single_trigger": "llm-async/{model}/v1/chat/completions",
            "batch_init": "llm-async/{model}/v1/chat/completions/batch",
            "batch_upload": "llm-async/{model}/v1/chat/completions/batch/{id}",
            "batch_submit": "llm-async/{model}/v1/batch/{id}/submit",
            "batch_cancel": "llm-async/{model}/v1/batch/{id}/cancel",
            "task_stop": "llm-async/{model}/v1/{id}/stop",
            "task_resume": "llm-async/{model}/v1/{id}/resume",
            "task_detail": "llm-async/{model}/v1/{id}",
            "result_page": "llm-async/{model}/v1/result/page/{id}",
        }

    def _build_url(self, path_key: str, **kwargs: str) -> str:
        template = self.api_paths[path_key]
        path = template.format(**kwargs)
        return urljoin(f"{self.host}/", path)

    def _request(
        self,
        method: str,
        url: str,
        data: dict[str, Any] | list[Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    json=data,
                    headers=headers,
                    timeout=self.timeout,
                )
                if not 200 <= response.status_code < 300:
                    try:
                        detail: Any = response.json()
                    except ValueError:
                        detail = response.text[:500]
                    raise AsyncLLMError(f"HTTP {response.status_code}: {detail}")
                payload = response.json()
                if not isinstance(payload, dict):
                    raise AsyncLLMError("platform response must be a JSON object")
                if payload.get("returnCode") != "SUC0000":
                    raise AsyncLLMError(str(payload.get("errorMsg") or payload))
                return payload
            except (requests.RequestException, ValueError, AsyncLLMError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(self.retry_sleep)
        raise AsyncLLMError(str(last_error))

    @staticmethod
    def _body(payload: dict[str, Any]) -> dict[str, Any]:
        body = payload.get("body", {})
        if not isinstance(body, dict):
            raise AsyncLLMError("platform response body must be a JSON object")
        return body

    def trigger_single_task(
        self,
        content: AsyncTaskContent,
        job_name: str | None = None,
        callback: CallbackConfig | None = None,
    ) -> str:
        body = content.model_dump(mode="json", exclude_none=True)
        if callback is not None:
            body["metadata"] = {"callback": callback.model_dump(mode="json", exclude_none=True)}
        headers = {"x-job-name": job_name} if job_name else None
        payload = self._request(
            "POST",
            self._build_url("single_trigger", model=self.model),
            data=body,
            headers=headers,
        )
        task_id = self._body(payload).get("id")
        if not task_id:
            raise AsyncLLMError("single-task response did not contain an id")
        return str(task_id)

    def init_batch_task(
        self,
        job_name: str | None = None,
        callback: CallbackConfig | None = None,
    ) -> str:
        body = None
        if callback is not None:
            body = {"metadata": {"callback": callback.model_dump(mode="json", exclude_none=True)}}
        headers = {"x-job-name": job_name} if job_name else None
        payload = self._request(
            "POST",
            self._build_url("batch_init", model=self.model),
            data=body,
            headers=headers,
        )
        task_id = self._body(payload).get("id")
        if not task_id:
            raise AsyncLLMError("batch-init response did not contain an id")
        return str(task_id)

    def upload_batch_content(
        self,
        task_id: str,
        contents: list[AsyncTaskContent],
        params: AsyncTaskParams | None = None,
    ) -> None:
        serialized = [item.model_dump(mode="json", exclude_none=True) for item in contents]
        if params is not None:
            body: dict[str, Any] | list[Any] = {
                "params": params.model_dump(mode="json", exclude_none=True),
                "contents": serialized,
            }
        elif len(serialized) == 1:
            body = serialized[0]
        else:
            body = serialized
        self._request(
            "PUT",
            self._build_url("batch_upload", model=self.model, id=task_id),
            data=body,
        )

    def submit_batch_task(self, task_id: str) -> None:
        self._request("PUT", self._build_url("batch_submit", model=self.model, id=task_id))

    def cancel_batch_task(self, task_id: str) -> None:
        self._request("PUT", self._build_url("batch_cancel", model=self.model, id=task_id))

    def stop_task(self, task_id: str) -> None:
        self._request("PUT", self._build_url("task_stop", model=self.model, id=task_id))

    def resume_task(self, task_id: str) -> None:
        self._request("PUT", self._build_url("task_resume", model=self.model, id=task_id))

    def get_task_detail(self, task_id: str) -> AsyncTaskDetail:
        payload = self._request(
            "GET",
            self._build_url("task_detail", model=self.model, id=task_id),
        )
        return AsyncTaskDetail.model_validate(self._body(payload))

    def get_task_results(
        self,
        task_id: str,
        current_page: int = 1,
        page_size: int = 500,
    ) -> AsyncResultPage:
        payload = self._request(
            "POST",
            self._build_url("result_page", model=self.model, id=task_id),
            data={"currentPage": current_page, "pageSize": min(page_size, 500)},
        )
        return AsyncResultPage.model_validate(self._body(payload))

    def get_all_results(self, task_id: str) -> list[AsyncResultItem]:
        results: list[AsyncResultItem] = []
        current_page = 1
        while True:
            page = self.get_task_results(task_id, current_page=current_page, page_size=500)
            results.extend(page.data)
            if page.total_pages == 0 or current_page >= page.total_pages:
                return results
            current_page += 1

    def wait_for_completion(
        self,
        task_id: str,
        poll_interval: float = 10,
        max_wait_time: float = 3600,
    ) -> AsyncTaskDetail:
        start = time.monotonic()
        while True:
            detail = self.get_task_detail(task_id)
            if detail.state in TERMINAL_STATES:
                return detail
            if time.monotonic() - start > max_wait_time:
                raise AsyncLLMError(f"timed out waiting for task {task_id}")
            time.sleep(poll_interval)

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "CmbAsyncLLM":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
