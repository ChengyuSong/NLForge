"""LLM backend implementations."""

from .base import LLMBackend
from .claude import ClaudeBackend
from .gemini import GeminiBackend
from .llamacpp import LlamaCppBackend
from .ollama import OllamaBackend
from .openai import OpenAIBackend
from .pool import LLMPool

__all__ = [
    "LLMBackend",
    "ClaudeBackend",
    "GeminiBackend",
    "OpenAIBackend",
    "OllamaBackend",
    "LlamaCppBackend",
    "LLMPool",
    "build_backend_kwargs",
]


def build_backend_kwargs(
    backend: str,
    llm_host: str = "localhost",
    llm_port: int | None = None,
    disable_thinking: bool = False,
) -> dict:
    """Build kwargs dict for create_backend() from common CLI options."""
    kwargs: dict = {}
    if backend == "llamacpp":
        kwargs["host"] = llm_host
        kwargs["port"] = llm_port if llm_port is not None else 8080
    elif backend == "ollama":
        if llm_port is None:
            llm_port = 11434
        kwargs["base_url"] = f"http://{llm_host}:{llm_port}"
    if disable_thinking:
        kwargs["enable_thinking"] = False
    return kwargs


def create_backend(
    backend_type: str,
    model: str | None = None,
    **kwargs,
) -> LLMBackend:
    """
    Create an LLM backend instance.

    Args:
        backend_type: One of "claude", "openai", "ollama", "llamacpp", "gemini"
        model: Model name (uses default if not specified)
        **kwargs: Additional backend-specific arguments

    Returns:
        An LLMBackend instance
    """
    if backend_type == "claude":
        return ClaudeBackend(model=model, **kwargs)
    elif backend_type == "gemini":
        return GeminiBackend(model=model, **kwargs)
    elif backend_type == "openai":
        return OpenAIBackend(model=model, **kwargs)
    elif backend_type == "ollama":
        return OllamaBackend(model=model, **kwargs)
    elif backend_type == "llamacpp":
        return LlamaCppBackend(model=model, **kwargs)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
