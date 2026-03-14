"""Abstract base class for LLM backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    """Response from an LLM."""

    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    def __init__(self, model: str | None = None):
        self.model = model or self.default_model

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default model for this backend."""
        pass

    @abstractmethod
    def complete(
        self, prompt: str, system: str | None = None, cache_system: bool = False,
        response_format: dict | None = None,
    ) -> str:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: The user prompt
            system: Optional system message
            cache_system: If True, request caching of the system message (backend-specific)
            response_format: Optional response format constraint (e.g. JSON schema)

        Returns:
            The completion text
        """
        pass

    @abstractmethod
    def complete_with_metadata(
        self, prompt: str, system: str | None = None, cache_system: bool = False,
        response_format: dict | None = None,
    ) -> LLMResponse:
        """
        Generate a completion with metadata.

        Args:
            prompt: The user prompt
            system: Optional system message
            cache_system: If True, request caching of the system message (backend-specific)

        Returns:
            LLMResponse with content and metadata
        """
        pass

    def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | None = None,
    ) -> Any:
        """Generate a completion with tool use. Override in subclasses that support tools."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support tool use")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"
