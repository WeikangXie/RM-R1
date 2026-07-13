"""Business-stage settings for Content RM annotation."""

from infrastructure.config import env_int


FIRST_PASS_LLM_MAX_TOKENS = env_int("FIRST_PASS_LLM_MAX_TOKENS", 512)
SECOND_PASS_LLM_MAX_TOKENS = env_int("SECOND_PASS_LLM_MAX_TOKENS", 768)
