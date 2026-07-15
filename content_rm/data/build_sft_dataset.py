#!/usr/bin/env python3
"""Build SFT datasets from model annotations and sparse human overrides."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, TypeVar

CONTENT_RM_ROOT = Path(__file__).resolve().parents[1]
if str(CONTENT_RM_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTENT_RM_ROOT))

from infrastructure.common import read_model_jsonl, write_json, write_jsonl

from data.records import (
    FirstPassAnnotation,
    HumanReviewRecord,
    LLMAnnotationRecord,
    SecondPassRecord,
    decision_rubrics_are_consistent,
)
from data.rubric_utils import read_rubrics, rubrics_text

SYSTEM_PROMPT = """你是金融内容社区的 AI 回复审核助手。请严格依据给定 rubrics 判断 AI 回复是否可以通过运营审核。

输出必须是 JSON，不要输出 JSON 之外的文字。JSON 字段固定为：
{
  "violated_rubrics": ["命中的规则名称；如果没有则为空数组"],
  "reasoning": "简短说明为什么通过或不通过，必须引用具体内容依据",
  "decision": "pass 或 reject"
}

注意：
1. decision 只能是 pass 或 reject。
2. 若 AI 回复包含收益承诺、暗示确定收益、诱导买卖、个性化投资建议、事实不确定却说得过满等风险，应倾向 reject。
"""

T = TypeVar("T")


def unique_by_comment_id(rows: list[T], source_name: str) -> dict[str, T]:
    by_id: dict[str, T] = {}
    for row in rows:
        comment_id = str(getattr(row, "comment_id"))
        if comment_id in by_id:
            raise ValueError(f"duplicate comment_id in {source_name}: {comment_id}")
        by_id[comment_id] = row
    return by_id


def load_optional(path: Path | None, model_type: type[T]) -> list[T]:
    if path is None:
        return []
    if not path.exists():
        raise ValueError(f"input file does not exist: {path}")
    return read_model_jsonl(path, model_type)


def validate_rubrics(values: list[str], valid_rubrics: set[str], source: str) -> None:
    unknown = [value for value in values if value not in valid_rubrics]
    if unknown:
        raise ValueError(f"{source} contains unknown rubrics: {unknown}")


def user_prompt(record: LLMAnnotationRecord, rubrics: list[dict[str, str]]) -> str:
    context = record.context
    return f"""请审核下面这条金融内容社区 AI 回复。

Rubrics:
{rubrics_text(rubrics)}

业务上下文：
- 原帖/父评论：{context.text}
- 产品名称：{context.product_name}
- 话题：{", ".join(context.topic_titles)}
- 圈子：{", ".join(context.coterie_names)}
- 回复类型：{context.extend_type}

待审核 AI 回复：
{context.ai_reply}
"""


def choose_annotation(
    first: LLMAnnotationRecord,
    second: SecondPassRecord | None,
    human: HumanReviewRecord | None,
    valid_rubrics: set[str],
) -> tuple[FirstPassAnnotation | None, str]:
    comment_id = str(first.comment_id)
    if human is not None:
        validate_rubrics(human.violated_rubrics, valid_rubrics, f"human_review {comment_id}")
        if not decision_rubrics_are_consistent(
            first.audit_label, human.violated_rubrics
        ):
            raise ValueError(
                f"human_review {comment_id} has inconsistent decision and rubrics"
            )
        return FirstPassAnnotation(
            violated_rubrics=human.violated_rubrics,
            reasoning=human.reasoning,
            decision=first.audit_label,
        ), "human_review"

    if not first.ok or first.annotation is None:
        return None, "first_pass_failed"

    validate_rubrics(
        first.annotation.violated_rubrics, valid_rubrics, f"llm_annotations {comment_id}"
    )
    if first.annotation.decision == first.audit_label:
        # Model-generated supervision that contradicts its own decision is
        # counted and skipped instead of silently entering the SFT dataset.
        if not decision_rubrics_are_consistent(
            first.annotation.decision, first.annotation.violated_rubrics
        ):
            return None, "first_pass_inconsistent_decision_rubrics"
        return first.annotation, "first_pass"

    if second is None:
        return None, "unresolved_disagreement_missing_second_pass"
    if not second.ok or second.annotation is None:
        return None, "unresolved_disagreement_second_pass_failed"
    validate_rubrics(
        second.annotation.violated_rubrics,
        valid_rubrics,
        f"second_pass_annotations {comment_id}",
    )
    if second.annotation.status != "ok":
        return None, "unresolved_disagreement_need_review"
    if second.annotation.decision != first.audit_label:
        return None, "unresolved_disagreement_second_pass_decision_mismatch"
    if not decision_rubrics_are_consistent(
        second.annotation.decision, second.annotation.violated_rubrics
    ):
        return (
            None,
            "unresolved_disagreement_second_pass_inconsistent_decision_rubrics",
        )
    return FirstPassAnnotation(
        violated_rubrics=second.annotation.violated_rubrics,
        reasoning=second.annotation.reasoning,
        decision=first.audit_label,
    ), "second_pass"


def build_sft_rows(
    first_pass: list[LLMAnnotationRecord],
    second_by_id: dict[str, SecondPassRecord],
    human_by_id: dict[str, HumanReviewRecord],
    rubrics: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    valid_rubrics = {item["name"] for item in rubrics}
    first_ids = {str(record.comment_id) for record in first_pass}
    orphan_second = sorted(set(second_by_id) - first_ids)
    orphan_human = sorted(set(human_by_id) - first_ids)
    if orphan_second:
        raise ValueError(f"orphan second-pass comment_id values: {orphan_second}")
    if orphan_human:
        raise ValueError(f"orphan human-review comment_id values: {orphan_human}")

    rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    for first in first_pass:
        comment_id = str(first.comment_id)
        annotation, source = choose_annotation(
            first,
            second_by_id.get(comment_id),
            human_by_id.get(comment_id),
            valid_rubrics,
        )
        stats[source] += 1
        if annotation is None:
            continue
        response = annotation.model_copy(update={"decision": first.audit_label})
        rows.append(
            {
                "comment_id": comment_id,
                "context_messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt(first, rubrics)},
                ],
                "response": json.dumps(response.model_dump(mode="json"), ensure_ascii=False),
                "audit_label": first.audit_label,
            }
        )
    return rows, stats


def stratified_split(
    rows: list[dict[str, Any]], test_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[row["audit_label"]].append(row)

    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for label_rows in by_label.values():
        rng.shuffle(label_rows)
        if len(label_rows) <= 1 or test_ratio <= 0:
            test_count = 0
        else:
            test_count = min(max(1, round(len(label_rows) * test_ratio)), len(label_rows) - 1)
        test.extend(label_rows[:test_count])
        train.extend(label_rows[test_count:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def make_summary(
    first_pass: list[LLMAnnotationRecord],
    sft_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    stats: Counter[str],
    output_format: str,
    outputs: dict[str, str],
) -> dict[str, Any]:
    return {
        "output_format": output_format,
        "total_llm_annotation_rows": len(first_pass),
        "usable_sft_rows": len(sft_rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "annotation_source_counts": dict(stats),
        "label_counts": dict(Counter(row["audit_label"] for row in sft_rows)),
        "train_label_counts": dict(Counter(row["audit_label"] for row in train_rows)),
        "test_label_counts": dict(Counter(row["audit_label"] for row in test_rows)),
        "outputs": outputs,
    }


def extract_messages(row: dict[str, Any]) -> tuple[str, str]:
    system = ""
    prompt = ""
    for message in row.get("context_messages", []):
        if message.get("role") == "system" and not system:
            system = str(message.get("content", ""))
        elif message.get("role") == "user" and not prompt:
            prompt = str(message.get("content", ""))
    return system, prompt


def to_llamafactory_alpaca(row: dict[str, Any]) -> dict[str, str]:
    system, prompt = extract_messages(row)
    return {
        "instruction": prompt,
        "input": "",
        "output": str(row.get("response", "")),
        "system": system,
    }


def to_post_train_platform(row: dict[str, Any]) -> dict[str, str]:
    system, prompt = extract_messages(row)
    return {"system": system, "prompt": prompt, "response": str(row.get("response", ""))}


def write_outputs(
    output_dir: Path,
    first_pass: list[LLMAnnotationRecord],
    sft_rows: list[dict[str, Any]],
    stats: Counter[str],
    test_ratio: float,
    seed: int,
    output_format: str,
) -> dict[str, Any]:
    if output_format == "post_train_platform":
        backend_dir = output_dir / "post-train-platform"
        backend_dir.mkdir(parents=True, exist_ok=True)
        all_path = backend_dir / "all.jsonl"
        write_jsonl(all_path, [to_post_train_platform(row) for row in sft_rows])
        summary = make_summary(
            first_pass,
            sft_rows,
            sft_rows,
            [],
            stats,
            output_format,
            {"all": str(all_path)},
        )
        write_json(backend_dir / "summary.json", summary)
        return summary

    train_rows, test_rows = stratified_split(sft_rows, test_ratio, seed)
    if output_format == "llamafactory_alpaca":
        backend_dir = output_dir / "llamafactory_alpaca"
        backend_dir.mkdir(parents=True, exist_ok=True)
        train_path = backend_dir / "train.json"
        test_path = backend_dir / "test.json"
        info_path = backend_dir / "dataset_info.json"
        write_json(train_path, [to_llamafactory_alpaca(row) for row in train_rows])
        write_json(test_path, [to_llamafactory_alpaca(row) for row in test_rows])
        write_json(
            info_path,
            {
                "content_rm_train": {
                    "file_name": "train.json",
                    "columns": {
                        "prompt": "instruction",
                        "query": "input",
                        "response": "output",
                        "system": "system",
                    },
                },
                "content_rm_test": {
                    "file_name": "test.json",
                    "columns": {
                        "prompt": "instruction",
                        "query": "input",
                        "response": "output",
                        "system": "system",
                    },
                },
            },
        )
        outputs = {"train": str(train_path), "test": str(test_path), "dataset_info": str(info_path)}
    else:
        backend_dir = output_dir / "openrlhf"
        backend_dir.mkdir(parents=True, exist_ok=True)
        train_path = backend_dir / "train.jsonl"
        test_path = backend_dir / "test.jsonl"
        write_jsonl(train_path, train_rows)
        write_jsonl(test_path, test_rows)
        outputs = {"train": str(train_path), "test": str(test_path)}

    summary = make_summary(
        first_pass,
        sft_rows,
        train_rows,
        test_rows,
        stats,
        output_format,
        outputs,
    )
    write_json(backend_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reviewed Content RM SFT datasets.")
    parser.add_argument("--llm-annotations", type=Path, required=True)
    parser.add_argument("--second-pass-annotations", type=Path)
    parser.add_argument("--human-review", type=Path)
    parser.add_argument("--rubrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--write-llamafactory-alpaca", action="store_true")
    output_group.add_argument("--write-post-train-platform", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    first_pass = read_model_jsonl(args.llm_annotations, LLMAnnotationRecord)
    first_by_id = unique_by_comment_id(first_pass, "llm_annotations")
    if len(first_by_id) != len(first_pass):
        raise AssertionError("duplicate check failed")
    second = load_optional(args.second_pass_annotations, SecondPassRecord)
    human = load_optional(args.human_review, HumanReviewRecord)
    second_by_id = unique_by_comment_id(second, "second_pass_annotations")
    human_by_id = unique_by_comment_id(human, "human_review")
    rubrics = read_rubrics(args.rubrics)
    sft_rows, stats = build_sft_rows(first_pass, second_by_id, human_by_id, rubrics)

    if args.write_post_train_platform:
        output_format = "post_train_platform"
    elif args.write_llamafactory_alpaca:
        output_format = "llamafactory_alpaca"
    else:
        output_format = "openrlhf"
    summary = write_outputs(
        args.output_dir,
        first_pass,
        sft_rows,
        stats,
        args.test_ratio,
        args.seed,
        output_format,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
