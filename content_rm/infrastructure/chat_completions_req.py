"""Pydantic request models shared by company chat-completions clients."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base model for strict infrastructure contracts."""

    model_config = ConfigDict(extra="forbid")


class Message(StrictModel):
    role: str
    content: str


class ChatCompletionsReq(StrictModel):
    model: str | None = None
    messages: list[Message] = Field(default_factory=list)
    frequency_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    logprobs: bool | None = None
    max_tokens: int | None = None
    n: int | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    stop: list[str] = Field(default_factory=list)
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    user: str | None = None
    top_logprobs: int | None = None
    best_of: int | None = None
