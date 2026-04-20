"""Backward-compatible execution module exports."""

from multimodal_agent.core.execution import SAFE_BUILTINS, run_python_code

__all__ = ["SAFE_BUILTINS", "run_python_code"]
