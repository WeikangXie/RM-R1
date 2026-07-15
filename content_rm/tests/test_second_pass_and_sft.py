from __future__ import annotations

import json
from uuid import UUID

import pytest

from data.build_sft_dataset import build_sft_rows, unique_by_comment_id
from data.records import (
    FirstPassAnnotation,
    HumanReviewRecord,
    LLMAnnotationRecord,
    ReviewContext,
    SecondPassAnnotation,
    SecondPassRecord,
)
from data.second_pass_review import (
    build_tasks,
    parse_response_record as parse_second_pass_response,
)


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
    system_prompt = tasks[0].messages[0].content
    user_prompt = tasks[0].messages[1].content
    assert "待验证的候选 decision" in user_prompt
    assert "reject" in user_prompt
    assert "第一次 LLM 审核结果" not in user_prompt
    assert "first-2" not in user_prompt
    assert "可独立使用" in system_prompt
    assert "运营最终审核结果" not in system_prompt
    assert "真实审核判断" in system_prompt
    assert "不得为了等于候选 decision" in system_prompt


def test_second_pass_canonicalizes_known_rubric_alias() -> None:
    task = build_tasks(
        [first_record(1, "reject", "pass")],
        [{"name": "不得暗示收益", "description": "不得暗示确定收益"}],
    )[0]
    response = {
        "choices": [
            {
                "message": {
                    "content": '{"status": "ok", "violated_rubrics": ["暗示收益"], "reasoning": "存在收益暗示", "decision": "reject"}'
                }
            }
        ]
    }

    result = parse_second_pass_response(task, response, {"不得暗示收益"})

    assert result.ok
    assert result.annotation is not None
    assert result.annotation.violated_rubrics == ["不得暗示收益"]


def test_second_pass_decision_mismatch_becomes_need_review() -> None:
    task = build_tasks([first_record(1, "reject", "pass")], RUBRICS)[0]
    response = {
        "choices": [
            {
                "message": {
                    "content": '{"status": "ok", "violated_rubrics": [], "reasoning": "未发现违规，可以通过", "decision": "pass"}'
                }
            }
        ]
    }

    result = parse_second_pass_response(task, response, {"事实准确性"})

    assert result.ok
    assert result.annotation is not None
    assert result.annotation.status == "need_review"
    assert result.annotation.decision == "pass"


@pytest.mark.parametrize(
    ("audit_label", "first_decision", "decision", "violated_rubrics"),
    [
        ("reject", "pass", "reject", []),
        ("pass", "reject", "pass", ["事实准确性"]),
    ],
)
def test_second_pass_inconsistent_decision_rubrics_becomes_need_review(
    audit_label: str,
    first_decision: str,
    decision: str,
    violated_rubrics: list[str],
) -> None:
    task = build_tasks([first_record(1, audit_label, first_decision)], RUBRICS)[0]
    response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "status": "ok",
                            "violated_rubrics": violated_rubrics,
                            "reasoning": "结构不一致",
                            "decision": decision,
                        },
                        ensure_ascii=False,
                    )
                }
            }
        ]
    }

    result = parse_second_pass_response(task, response, {"事实准确性"})

    assert result.ok
    assert result.annotation is not None
    assert result.annotation.status == "need_review"


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


def test_sft_skips_inconsistent_model_supervision() -> None:
    inconsistent_first = LLMAnnotationRecord(
        comment_id=UUID(int=1),
        audit_label="pass",
        context=ReviewContext(text="text-1", ai_reply="reply-1"),
        ok=True,
        annotation=FirstPassAnnotation(
            violated_rubrics=["事实准确性"],
            reasoning="inconsistent first",
            decision="pass",
        ),
    )
    disagreement = first_record(2, "reject", "pass")
    inconsistent_second = SecondPassRecord(
        comment_id=UUID(int=2),
        ok=True,
        annotation=SecondPassAnnotation(
            status="ok",
            violated_rubrics=[],
            reasoning="inconsistent second",
            decision="reject",
        ),
    )

    rows, stats = build_sft_rows(
        [inconsistent_first, disagreement],
        {str(UUID(int=2)): inconsistent_second},
        {},
        RUBRICS,
    )

    assert not rows
    assert stats == {
        "first_pass_inconsistent_decision_rubrics": 1,
        "unresolved_disagreement_second_pass_inconsistent_decision_rubrics": 1,
    }


def test_sft_skips_status_ok_second_pass_decision_mismatch() -> None:
    disagreement = first_record(1, "reject", "pass")
    malformed_second = SecondPassRecord(
        comment_id=UUID(int=1),
        ok=True,
        annotation=SecondPassAnnotation(
            status="ok",
            violated_rubrics=[],
            reasoning="still pass",
            decision="pass",
        ),
    )

    rows, stats = build_sft_rows(
        [disagreement],
        {str(UUID(int=1)): malformed_second},
        {},
        RUBRICS,
    )

    assert not rows
    assert stats == {
        "unresolved_disagreement_second_pass_decision_mismatch": 1
    }


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
