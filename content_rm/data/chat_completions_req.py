"""Pydantic request models for the company chat-completions APIs."""

from __future__ import annotations

from pydantic import Field

from records import Message, StrictModel


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
