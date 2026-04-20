"""Model wrappers."""

from .chatglm_module import ChatGLMAnswerGenerator, ChatGLMConfig, generate_answer

__all__ = ["ChatGLMAnswerGenerator", "ChatGLMConfig", "generate_answer"]
