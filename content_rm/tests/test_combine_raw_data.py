from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

from data.combine_raw_data import COMMENT_FIELDS_TO_KEEP, FIELDS_TO_MERGE, DataCombiner


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def fake_comment(
    index: int,
    *,
    parent_id: str,
    state: str,
    extend_type: str,
    audit_state: str = "OPERATOR_AUDITED",
) -> dict:
    return {
        "commentId": str(UUID(int=index)),
        "commentContent": f"合成回复-{index}",
        "commentState": state,
        "extendType": extend_type,
        "commentType": "COMMENT",
        "parentType": "POST",
        "parentId": parent_id,
        "auditState": audit_state,
        "operatorName": "不应进入输出",
        "remark": "不应进入输出",
    }


def fake_post(index: int, *, product_name: str) -> dict:
    post_id = str(UUID(int=10_000 + index))
    return {
        "postId": post_id,
        "text": f"合成帖子-{index}",
        "productName": product_name,
        "topicList": [{"title": f"合成话题-{index}"}],
        "relatedCoterieList": [{"coterieName": f"合成圈子-{index}"}],
        "auditStatus": "PASS",
        "_id": post_id,
    }


def test_combiner_uses_whitelist_and_reports_distributions(tmp_path: Path) -> None:
    first_post = fake_post(1, product_name="合成产品-A")
    second_post = fake_post(2, product_name="合成产品-B")
    missing_post_id = str(UUID(int=99_999))
    comments = [
        fake_comment(
            1,
            parent_id=first_post["postId"],
            state="PUBLISHED",
            extend_type="AI_CHECK_IN",
        ),
        fake_comment(
            2,
            parent_id=first_post["postId"],
            state="HIDE",
            extend_type="AI_CHECK_IN",
        ),
        fake_comment(
            3,
            parent_id=second_post["postId"],
            state="PUBLISHED",
            extend_type="AI_OPINION_SHARE",
        ),
        fake_comment(
            4,
            parent_id=second_post["postId"],
            state="PUBLISHED",
            extend_type="AI_OPINION_SHARE",
            audit_state="NOT_AUDIT",
        ),
        fake_comment(
            5,
            parent_id=missing_post_id,
            state="HIDE",
            extend_type="AI_CHECK_IN",
        ),
    ]
    write_rows(tmp_path / "oa_export_test_comment.jsonl", comments)
    write_rows(tmp_path / "oa_export_test_post.jsonl", [first_post, second_post])

    combiner = DataCombiner(tmp_path)
    combiner.load_post_data()
    combiner.merge_data()
    summary = combiner.filter_merge_data()

    output_rows = read_rows(combiner.filtered_comment_file)
    assert len(output_rows) == 3
    assert summary == {
        "total": 3,
        "comment_state_counts": {"PUBLISHED": 2, "HIDE": 1},
        "audit_label_counts": {"pass": 2, "reject": 1},
        "extend_type_counts": {"AI_CHECK_IN": 2, "AI_OPINION_SHARE": 1},
        "extend_type_audit_label_counts": {
            "AI_CHECK_IN": {"pass": 1, "reject": 1},
            "AI_OPINION_SHARE": {"pass": 1},
        },
        "distinct_product_count": 2,
    }
    expected_fields = set(COMMENT_FIELDS_TO_KEEP) | set(FIELDS_TO_MERGE)
    assert all(set(row) == expected_fields for row in output_rows)
    assert all("auditState" not in row for row in output_rows)
    assert all("parentId" not in row for row in output_rows)
    assert len(read_rows(combiner.dirty_file)) == 1


def test_filter_merge_data_does_not_sample(tmp_path: Path) -> None:
    combiner = DataCombiner(tmp_path)
    rows = [
        {
            "commentId": str(UUID(int=index + 1)),
            "commentContent": "合成回复",
            "commentState": "PUBLISHED",
            "extendType": "AI_CHECK_IN",
        }
        for index in range(501)
    ]
    write_rows(combiner.combine_file, rows)

    summary = combiner.filter_merge_data()

    assert summary["total"] == 501
    assert summary["audit_label_counts"] == {"pass": 501}
    assert len(read_rows(combiner.filtered_comment_file)) == 501
