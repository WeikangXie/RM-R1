from __future__ import annotations

import json
from uuid import UUID

import pytest

from build_sft_dataset import build_sft_rows, unique_by_comment_id
from records import (
    FirstPassAnnotation,
    HumanReviewRecord,
    LLMAnnotationRecord,
    ReviewContext,
    SecondPassAnnotation,
    SecondPassRecord,
)
from second_pass_review import build_tasks


RUBRICS = [{"name": "事实准确性", "description": "事实必须准确"}]


def first_record(index: int, audit: str, decision: str | None) -> LLMAnnotationRecord:
    comment_id = UUID(int=index)
    context = ReviewContext(text=f"text-{index}", ai_reply=f"reply-{index}")
    if decision is None:
        return LLMAnnotationRecord(
            comment_id=comment_id,
            audit_label=audit,
            context=context,
            ok=False,
            error="failed",
        )
    return LLMAnnotationRecord(
        comment_id=comment_id,
        audit_label=audit,
        context=context,
        ok=True,
        annotation=FirstPassAnnotation(
            violated_rubrics=[] if decision == "pass" else ["事实准确性"],
            reasoning=f"first-{index}",
            decision=decision,
        ),
        raw_content="{}",
    )


def test_second_pass_selects_only_successful_disagreements() -> None:
    records = [
        first_record(1, "pass", "pass"),
        first_record(2, "reject", "pass"),
        first_record(3, "reject", None),
    ]
    tasks = build_tasks(records, RUBRICS)
    assert [str(task.comment_id) for task in tasks] == [str(UUID(int=2))]
    assert tasks[0].first_annotation is not None


def test_sft_precedence_and_unresolved_disagreement_skip() -> None:
    first = [
        first_record(1, "pass", "pass"),
        first_record(2, "reject", "pass"),
        first_record(3, "reject", "pass"),
        first_record(4, "reject", None),
    ]
    second = {
        str(UUID(int=2)): SecondPassRecord(
            comment_id=UUID(int=2),
            ok=True,
            annotation=SecondPassAnnotation(
                status="ok",
                violated_rubrics=["事实准确性"],
                reasoning="second-2",
                decision="reject",
            ),
        ),
        str(UUID(int=3)): SecondPassRecord(
            comment_id=UUID(int=3),
            ok=True,
            annotation=SecondPassAnnotation(
                status="need_review",
                violated_rubrics=[],
                reasoning="cannot explain",
                decision="reject",
            ),
        ),
    }
    human = {
        str(UUID(int=4)): HumanReviewRecord(
            comment_id=UUID(int=4),
            violated_rubrics=["事实准确性"],
            reasoning="human-4",
        )
    }
    rows, stats = build_sft_rows(first, second, human, RUBRICS)
    assert len(rows) == 3
    assert stats == {
        "first_pass": 1,
        "second_pass": 1,
        "unresolved_disagreement_need_review": 1,
        "human_review": 1,
    }
    responses = {row["comment_id"]: json.loads(row["response"]) for row in rows}
    assert responses[str(UUID(int=1))]["reasoning"] == "first-1"
    assert responses[str(UUID(int=2))]["reasoning"] == "second-2"
    assert responses[str(UUID(int=4))]["reasoning"] == "human-4"
    assert responses[str(UUID(int=2))]["decision"] == "reject"
    assert all("sample_id" not in row for row in rows)


def test_sft_rejects_orphan_human_review() -> None:
    first = [first_record(1, "pass", "pass")]
    human = HumanReviewRecord(
        comment_id=UUID(int=2), violated_rubrics=[], reasoning="human"
    )
    with pytest.raises(ValueError, match="orphan human-review"):
        build_sft_rows(first, {}, {str(human.comment_id): human}, RUBRICS)


def test_duplicate_comment_id_is_rejected() -> None:
    row = first_record(1, "pass", "pass")
    with pytest.raises(ValueError, match="duplicate comment_id"):
        unique_by_comment_id([row, row], "llm_annotations")
