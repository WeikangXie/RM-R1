#!/usr/bin/env python3
"""Local defaults for company LLM calls used by data scripts.

Prefer setting LLM_AUTHORIZATION in the environment instead of writing tokens
into this tracked file.
"""

from __future__ import annotations

import os


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def env_optional_int(name: str) -> int | None:
    value = os.getenv(name)
    return int(value) if value else None


LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_AUTHORIZATION = os.getenv("LLM_AUTHORIZATION", "")

LLM_TEMPERATURE = env_float("LLM_TEMPERATURE", 0.0)
LLM_TOP_P = env_float("LLM_TOP_P", 1.0)
LLM_RESPONSE_FORMAT = os.getenv("LLM_RESPONSE_FORMAT", "json_object")
LLM_SEED = env_optional_int("LLM_SEED")
LLM_TIMEOUT = env_float("LLM_TIMEOUT", 60.0)
LLM_RETRIES = env_int("LLM_RETRIES", 2)
LLM_RETRY_SLEEP = env_float("LLM_RETRY_SLEEP", 1.0)

FIRST_PASS_LLM_MAX_TOKENS = env_int("FIRST_PASS_LLM_MAX_TOKENS", 512)
SECOND_PASS_LLM_MAX_TOKENS = env_int("SECOND_PASS_LLM_MAX_TOKENS", 768)
