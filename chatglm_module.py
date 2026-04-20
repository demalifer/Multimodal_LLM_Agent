"""Backward-compatible ChatGLM exports."""

from multimodal_agent.models.chatglm_module import (
    ChatGLMAnswerGenerator,
    ChatGLMConfig,
    generate_answer,
)

__all__ = ["ChatGLMAnswerGenerator", "ChatGLMConfig", "generate_answer"]
