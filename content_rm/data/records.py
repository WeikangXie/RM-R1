"""Validated business records for the Content RM data pipeline."""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from infrastructure.chat_completions_req import Message, StrictModel

Label = Literal["pass", "reject"]
ReviewStatus = Literal["ok", "need_review"]


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


class AnnotationTask(StrictModel):
    comment_id: UUID
    audit_label: Label
    context: ReviewContext
    messages: list[Message]
    first_annotation: FirstPassAnnotation | None = None
