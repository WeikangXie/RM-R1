from __future__ import annotations

from typing import Any

from cmb_async_model import AsyncTaskContent, AsyncTaskParams, CmbAsyncLLM
from records import Message


class FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeSession:
    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.calls: list[dict[str, Any]] = []
        self.responses = [
            FakeResponse({"returnCode": "SUC0000", "body": {"id": "job-1"}}),
            FakeResponse({"returnCode": "SUC0000", "body": {}}),
        ]

    def request(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        return self.responses.pop(0)

    def close(self) -> None:
        pass


def test_client_uses_exact_authorization_and_keeps_zero_parameters() -> None:
    session = FakeSession()
    client = CmbAsyncLLM(
        host="https://platform.example/base",
        authorization="Bearer exact-token",
        model="model-a",
        session=session,  # type: ignore[arg-type]
    )
    assert session.headers["Authorization"] == "Bearer exact-token"
    task_id = client.init_batch_task(job_name="job")
    client.upload_batch_content(
        task_id,
        [
            AsyncTaskContent(
                custom_id="12345678-1234-5678-1234-567812345678",
                messages=[Message(role="user", content="prompt")],
                model="model-a",
            )
        ],
        AsyncTaskParams(temperature=0.0, max_tokens=100),
    )
    assert session.calls[0]["headers"] == {"x-job-name": "job"}
    upload_body = session.calls[1]["json"]
    assert upload_body["params"]["temperature"] == 0.0
    assert upload_body["contents"][0]["custom_id"] == "12345678-1234-5678-1234-567812345678"
    assert "response_format" not in str(upload_body)
