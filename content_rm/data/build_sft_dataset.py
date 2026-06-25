#!/usr/bin/env python3
"""Build final SFT JSONL files from reviewed domain annotations."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VALID_LABELS = {"pass", "reject"}


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no} must be a JSON object")
            rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def text_or_empty(value: Any) -> str:
    return value if isinstance(value, str) else ""


def read_rubrics(path: Path) -> list[dict[str, str]]:
    rubrics: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("|"):
                continue
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) < 2:
                continue
            name, description = cells[0], cells[1]
            if name in {"Rubric", "-----------"} or set(name) <= {"-"}:
                continue
            rubrics.append({"name": name, "description": description})
    if not rubrics:
        raise ValueError(f"No rubrics parsed from {path}")
    return rubrics


def rubrics_text(rubrics: list[dict[str, str]]) -> str:
    return "\n".join(f"- {r['name']}: {r['description']}" for r in rubrics)


def parse_rubrics(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if not isinstance(value, str) or not value.strip():
        return []
    if "|" in value:
        parts = value.split("|")
    elif "," in value:
        parts = value.split(",")
    else:
        parts = [value]
    return [part.strip() for part in parts if part.strip()]


def load_llm_annotations(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    by_id: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if not row.get("ok") or not isinstance(row.get("annotation"), dict):
            continue
        custom_id = text_or_empty(row.get("custom_id"))
        if custom_id:
            by_id[custom_id] = row["annotation"]
    return by_id


def user_prompt(review: dict[str, Any], rubrics: list[dict[str, str]]) -> str:
    return f"""请审核下面这条金融内容社区 AI 回复。

Rubrics:
{rubrics_text(rubrics)}

业务上下文：
- 原帖/父评论：{text_or_empty(review.get("text"))}
- 产品名称：{text_or_empty(review.get("product_name"))}
- 话题：{text_or_empty(review.get("topic_titles"))}
- 圈子：{text_or_empty(review.get("coterie_names"))}
- 回复类型：{text_or_empty(review.get("extend_type"))}

待审核 AI 回复：
{text_or_empty(review.get("ai_reply"))}
"""


def make_response(
    audit_label: str,
    human_review: dict[str, Any],
    llm_annotation: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, str]:
    human_reasoning = text_or_empty(human_review.get("human_reasoning")).strip()
    llm_reasoning = text_or_empty(human_review.get("llm_reasoning")).strip()
    if not llm_reasoning and llm_annotation:
        llm_reasoning = text_or_empty(llm_annotation.get("reasoning")).strip()
    reasoning = human_reasoning or llm_reasoning

    rubrics = parse_rubrics(human_review.get("violated_rubrics"))
    if not rubrics and llm_annotation:
        rubrics = parse_rubrics(llm_annotation.get("violated_rubrics"))

    if not reasoning:
        return None, "missing_reasoning"
    if audit_label not in VALID_LABELS:
        return None, "invalid_audit_label"

    # The operator label remains the final supervised decision. LLM output is
    # used only to provide a candidate explanation for human review.
    return {
        "violated_rubrics": rubrics,
        "reasoning": reasoning,
        "decision": audit_label,
    }, "ok"


def build_from_human_review(
    human_review_rows: list[dict[str, Any]],
    llm_annotations: dict[str, dict[str, Any]],
    rubrics: list[dict[str, str]],
    allow_empty_reasoning: bool,
) -> tuple[list[dict[str, Any]], Counter]:
    rows: list[dict[str, Any]] = []
    stats: Counter = Counter()

    for review in human_review_rows:
        sample_id = text_or_empty(review.get("sample_id"))
        audit_label = text_or_empty(review.get("audit_label"))
        response_obj, status = make_response(audit_label, review, llm_annotations.get(sample_id))
        if response_obj is None and allow_empty_reasoning and status == "missing_reasoning" and audit_label in VALID_LABELS:
            response_obj = {
                "violated_rubrics": parse_rubrics(review.get("violated_rubrics")),
                "reasoning": "",
                "decision": audit_label,
            }
            status = "ok_empty_reasoning"
        if response_obj is None:
            stats[status] += 1
            continue

        rows.append(
            {
                "sample_id": sample_id,
                "context_messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt(review, rubrics)},
                ],
                "response": json.dumps(response_obj, ensure_ascii=False),
                "audit_label": audit_label,
            }
        )
        stats[status] += 1

    return rows, stats


def stratified_split(rows: list[dict[str, Any]], test_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[row["audit_label"]].append(row)

    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for label_rows in by_label.values():
        rng.shuffle(label_rows)
        if len(label_rows) <= 1 or test_ratio <= 0:
            split_count = 0
        else:
            split_count = max(1, round(len(label_rows) * test_ratio))
            split_count = min(split_count, len(label_rows) - 1)
        test_rows.extend(label_rows[:split_count])
        train_rows.extend(label_rows[split_count:])

    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    return train_rows, test_rows


def make_summary(
    human_review_rows: list[dict[str, Any]],
    sft_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    stats: Counter,
    output_format: str,
    outputs: dict[str, str],
) -> dict[str, Any]:
    return {
        "output_format": output_format,
        "total_review_rows": len(human_review_rows),
        "usable_sft_rows": len(sft_rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "build_stats": dict(stats),
        "label_counts": dict(Counter(row["audit_label"] for row in sft_rows)),
        "train_label_counts": dict(Counter(row["audit_label"] for row in train_rows)),
        "test_label_counts": dict(Counter(row["audit_label"] for row in test_rows)),
        "outputs": outputs,
    }


def to_llamafactory_alpaca(row: dict[str, Any]) -> dict[str, str]:
    context_messages = row.get("context_messages")
    system = ""
    instruction = ""
    if isinstance(context_messages, list):
        for message in context_messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = text_or_empty(message.get("content"))
            if role == "system" and not system:
                system = content
            elif role == "user" and not instruction:
                instruction = content

    return {
        "instruction": instruction,
        "input": "",
        "output": text_or_empty(row.get("response")),
        "system": system,
    }


def write_openrlhf_outputs(
    output_dir: Path,
    human_review_rows: list[dict[str, Any]],
    sft_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    stats: Counter,
) -> dict[str, Any]:
    backend_dir = output_dir / "openrlhf"
    backend_dir.mkdir(parents=True, exist_ok=True)
    train_path = backend_dir / "train.jsonl"
    test_path = backend_dir / "test.jsonl"
    summary_path = backend_dir / "summary.json"

    write_jsonl(train_path, train_rows)
    write_jsonl(test_path, test_rows)

    summary = make_summary(
        human_review_rows,
        sft_rows,
        train_rows,
        test_rows,
        stats,
        output_format="openrlhf",
        outputs={
            "train": str(train_path),
            "test": str(test_path),
        },
    )
    write_json(summary_path, summary)
    return summary


def write_llamafactory_alpaca_outputs(
    output_dir: Path,
    human_review_rows: list[dict[str, Any]],
    sft_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    stats: Counter,
) -> dict[str, Any]:
    backend_dir = output_dir / "llamafactory_alpaca"
    backend_dir.mkdir(parents=True, exist_ok=True)
    train_path = backend_dir / "train.json"
    test_path = backend_dir / "test.json"
    dataset_info_path = backend_dir / "dataset_info.json"
    summary_path = backend_dir / "summary.json"

    write_json(train_path, [to_llamafactory_alpaca(row) for row in train_rows])
    write_json(test_path, [to_llamafactory_alpaca(row) for row in test_rows])

    dataset_info = {
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
    }
    write_json(dataset_info_path, dataset_info)

    summary = make_summary(
        human_review_rows,
        sft_rows,
        train_rows,
        test_rows,
        stats,
        output_format="llamafactory_alpaca",
        outputs={
            "train": str(train_path),
            "test": str(test_path),
            "dataset_info": str(dataset_info_path),
        },
    )
    write_json(summary_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reviewed SFT train/test JSONL files.")
    parser.add_argument("--human-review", type=Path, required=True, help="Human review JSONL with reasoning fields.")
    parser.add_argument("--rubrics", type=Path, required=True, help="Rubrics markdown file.")
    parser.add_argument("--llm-annotations", type=Path, default=None, help="Optional llm_annotations.jsonl.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Parent output directory for backend-specific SFT files.")
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-empty-reasoning", action="store_true", help="For debugging only; not recommended for real SFT.")
    parser.add_argument(
        "--write-llamafactory-alpaca",
        action="store_true",
        help="Write only LLaMA-Factory Alpaca SFT files under llamafactory_alpaca/ instead of OpenRLHF JSONL.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    human_review_rows = read_jsonl(args.human_review)
    rubrics = read_rubrics(args.rubrics)
    llm_annotations = load_llm_annotations(args.llm_annotations)

    sft_rows, stats = build_from_human_review(
        human_review_rows,
        llm_annotations,
        rubrics,
        allow_empty_reasoning=args.allow_empty_reasoning,
    )
    train_rows, test_rows = stratified_split(sft_rows, args.test_ratio, args.seed)

    if args.write_llamafactory_alpaca:
        summary = write_llamafactory_alpaca_outputs(
            args.output_dir,
            human_review_rows,
            sft_rows,
            train_rows,
            test_rows,
            stats,
        )
    else:
        summary = write_openrlhf_outputs(
            args.output_dir,
            human_review_rows,
            sft_rows,
            train_rows,
            test_rows,
            stats,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
