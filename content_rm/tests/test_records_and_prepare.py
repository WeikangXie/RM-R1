from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import ValidationError

from data.prepare_dataset import build_tasks, parse_response_record, to_async_tasks
from data.records import LLMAnnotationRecord, RawComment


RUBRICS = [{"name": "事实准确性", "description": "事实必须准确"}]


def raw_row(comment_id: str) -> dict:
    return {
        "commentId": comment_id,
        "commentState": "PUBLISHED",
        "text": "<p>原帖</p>",
        "commentContent": "回复",
        "productName": "产品",
        "topicList": [{"title": "话题", "ignored": True}],
        "relatedCoterieList": [{"coterieName": "圈子"}],
        "unknownBusinessField": 123,
    }


def test_raw_comment_maps_comment_id_and_ignores_extra_fields() -> None:
    value = "12345678-1234-5678-1234-567812345678"
    record = RawComment.model_validate(raw_row(value))
    assert record.comment_id == UUID(value)
    assert record.topic_list[0].title == "话题"


def test_raw_comment_rejects_invalid_uuid() -> None:
    with pytest.raises(ValidationError):
        RawComment.model_validate(raw_row("not-a-uuid"))


def test_build_tasks_rejects_duplicate_comment_id_before_network() -> None:
    value = "12345678-1234-5678-1234-567812345678"
    with pytest.raises(ValueError, match="duplicate comment_id"):
        build_tasks([raw_row(value), raw_row(value)], RUBRICS)


def test_first_pass_response_uses_new_contract_only() -> None:
    task = build_tasks(
        [raw_row("12345678-1234-5678-1234-567812345678")], RUBRICS
    )[0]
    response = {
        "choices": [
            {
                "message": {
                    "content": '{"violated_rubrics": [], "reasoning": "未命中规则", "decision": "pass"}'
                }
            }
        ]
    }
    result = parse_response_record(task, response, {"事实准确性"})
    dumped = result.model_dump(mode="json")
    assert LLMAnnotationRecord.model_validate(dumped).ok
    assert dumped["comment_id"] == str(task.comment_id)
    assert "sample_id" not in dumped
    assert "custom_id" not in dumped


def test_async_adapter_uses_comment_id_as_custom_id() -> None:
    task = build_tasks(
        [raw_row("12345678-1234-5678-1234-567812345678")], RUBRICS
    )[0]
    async_task = to_async_tasks([task])[0]
    assert async_task.custom_id == str(task.comment_id)
    assert async_task.messages == task.messages


def test_first_pass_rejects_unknown_rubric() -> None:
    task = build_tasks(
        [raw_row("12345678-1234-5678-1234-567812345678")], RUBRICS
    )[0]
    response = {
        "choices": [
            {
                "message": {
                    "content": '{"violated_rubrics": ["不存在"], "reasoning": "命中", "decision": "reject"}'
                }
            }
        ]
    }
    result = parse_response_record(task, response, {"事实准确性"})
    assert not result.ok
    assert "unknown rubrics" in (result.error or "")
