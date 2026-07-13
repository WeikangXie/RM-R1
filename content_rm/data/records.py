"""Validated records shared by the Content RM data pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Label = Literal["pass", "reject"]
ReviewStatus = Literal["ok", "need_review"]
RunStage = Literal["first_pass", "second_pass"]
JobLifecycle = Literal["planned", "initialized", "uploaded", "submitted"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RawTopic(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: Any = ""


class RawCoterie(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    coterie_name: Any = Field(default="", alias="coterieName")


class RawComment(BaseModel):
    """The subset of company raw fields consumed by this project."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    comment_id: UUID = Field(alias="commentId")
    text: Any = ""
    parent_type: Any = Field(default="", alias="parentType")
    comment_content: Any = Field(default="", alias="commentContent")
    product_name: Any = Field(default="", alias="productName")
    extend_type: Any = Field(default="", alias="extendType")
    comment_type: Any = Field(default="", alias="commentType")
    comment_state: Any = Field(default="", alias="commentState")
    topic_list: list[RawTopic] = Field(default_factory=list, alias="topicList")
    related_coterie_list: list[RawCoterie] = Field(default_factory=list, alias="relatedCoterieList")

    @field_validator("topic_list", "related_coterie_list", mode="before")
    @classmethod
    def null_lists_are_empty(cls, value: Any) -> Any:
        return value if isinstance(value, list) else []


class ReviewContext(StrictModel):
    text: str = ""
    ai_reply: str = ""
    product_name: str = ""
    extend_type: str = ""
    comment_state: str = ""
    topic_titles: list[str] = Field(default_factory=list)
    coterie_names: list[str] = Field(default_factory=list)


def _normalize_rubrics(values: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = value.strip()
        if item and item not in seen:
            normalized.append(item)
            seen.add(item)
    return normalized


class FirstPassAnnotation(StrictModel):
    violated_rubrics: list[str] = Field(default_factory=list)
    reasoning: str
    decision: Label

    @field_validator("violated_rubrics")
    @classmethod
    def normalize_rubrics(cls, value: list[str]) -> list[str]:
        return _normalize_rubrics(value)

    @field_validator("reasoning")
    @classmethod
    def reasoning_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("reasoning must not be empty")
        return value


class LLMAnnotationRecord(StrictModel):
    comment_id: UUID
    audit_label: Label
    context: ReviewContext
    ok: bool
    annotation: FirstPassAnnotation | None = None
    raw_content: str | None = None
    error: str | None = None

    @model_validator(mode="after")
    def validate_result(self) -> "LLMAnnotationRecord":
        if self.ok and self.annotation is None:
            raise ValueError("successful annotation record requires annotation")
        if not self.ok and not self.error:
            raise ValueError("failed annotation record requires error")
        return self


class HumanReviewRecord(StrictModel):
    comment_id: UUID
    violated_rubrics: list[str] = Field(default_factory=list)
    reasoning: str
    review_note: str = ""

    @field_validator("violated_rubrics")
    @classmethod
    def normalize_rubrics(cls, value: list[str]) -> list[str]:
        return _normalize_rubrics(value)

    @field_validator("reasoning")
    @classmethod
    def reasoning_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("reasoning must not be empty")
        return value


class SecondPassAnnotation(StrictModel):
    status: ReviewStatus
    violated_rubrics: list[str] = Field(default_factory=list)
    reasoning: str
    decision: Label

    @field_validator("violated_rubrics")
    @classmethod
    def normalize_rubrics(cls, value: list[str]) -> list[str]:
        return _normalize_rubrics(value)

    @field_validator("reasoning")
    @classmethod
    def reasoning_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("reasoning must not be empty")
        return value


class SecondPassRecord(StrictModel):
    comment_id: UUID
    ok: bool
    annotation: SecondPassAnnotation | None = None
    raw_content: str | None = None
    error: str | None = None

    @model_validator(mode="after")
    def validate_result(self) -> "SecondPassRecord":
        if self.ok and self.annotation is None:
            raise ValueError("successful second-pass record requires annotation")
        if not self.ok and not self.error:
            raise ValueError("failed second-pass record requires error")
        return self


class Message(StrictModel):
    role: str
    content: str


class AnnotationTask(StrictModel):
    comment_id: UUID
    audit_label: Label
    context: ReviewContext
    messages: list[Message]
    first_annotation: FirstPassAnnotation | None = None


class AsyncRequestParams(StrictModel):
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    seed: int | None = None


class AsyncJobRecord(StrictModel):
    comment_ids: list[UUID]
    attempt: int = 1
    lifecycle: JobLifecycle = "planned"
    task_id: str | None = None
    platform_state: str | None = None
    total: int | None = None
    success: int | None = None
    failed: int | None = None
    error: str | None = None


class AsyncRunManifest(StrictModel):
    schema_version: int = 1
    stage: RunStage
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    input_path: str
    input_sha256: str
    rubrics_path: str
    rubrics_sha256: str
    output_path: str
    model: str
    batch_size: int
    params: AsyncRequestParams
    task_snapshot: str
    jobs: list[AsyncJobRecord] = Field(default_factory=list)
    failed_comment_ids: list[UUID] = Field(default_factory=list)

    @field_validator("batch_size")
    @classmethod
    def positive_batch_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("batch_size must be positive")
        return value


def model_rows(rows: list[BaseModel]) -> list[dict[str, Any]]:
    return [row.model_dump(mode="json") for row in rows]


def read_model_jsonl(path: Path, model_type: type[BaseModel]) -> list[Any]:
    from common import read_jsonl

    return [model_type.model_validate(row) for row in read_jsonl(path)]
