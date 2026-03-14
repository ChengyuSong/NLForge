"""Anthropic Claude backend (direct API and Vertex AI)."""

import os
from typing import Any

from .base import LLMBackend, LLMResponse


class ClaudeBackend(LLMBackend):
    """Anthropic Claude API backend with automatic Vertex AI support.

    Auto-detects client type:
    - If ANTHROPIC_API_KEY is set: uses anthropic.Anthropic
    - If GCP project env vars are set: uses anthropic.AnthropicVertex
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        project_id: str | None = None,
        location: str | None = None,
        max_tokens: int = 4096,
    ):
        super().__init__(model)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.project_id = (
            project_id
            or os.environ.get("VERTEX_AI_PROJECT")
            or os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("CLOUD_ML_PROJECT_ID")
        )
        self.location = location or os.environ.get("VERTEX_AI_LOCATION", "global")
        self.max_tokens = max_tokens
        self._client = None

    @property
    def default_model(self) -> str:
        return "claude-sonnet-4-6@default"

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package required. Install with: pip install anthropic"
                ) from None

            if self.api_key:
                self._client = anthropic.Anthropic(api_key=self.api_key)
            elif self.project_id:
                from anthropic import AnthropicVertex

                self._client = AnthropicVertex(
                    project_id=self.project_id,
                    region=self.location,
                )
            else:
                raise ValueError(
                    "Set ANTHROPIC_API_KEY or GOOGLE_CLOUD_PROJECT for Vertex AI."
                )
        return self._client

    def complete(
        self, prompt: str, system: str | None = None, cache_system: bool = False,
        response_format: dict | None = None,
    ) -> str:
        """Generate a completion using Claude."""
        response = self.complete_with_metadata(prompt, system, cache_system=cache_system)
        return response.content

    def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | None = None,
    ) -> Any:
        """Generate a completion with tool use support.

        Args:
            messages: List of message dicts with role and content
            tools: Optional list of tool definitions
            system: Optional system message

        Returns:
            Raw API response object with tool_use blocks if applicable
        """
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }

        if system:
            kwargs["system"] = system

        if tools:
            kwargs["tools"] = tools

        response = self.client.messages.create(**kwargs)

        if os.environ.get("CLAUDE_DEBUG"):
            print("[CLAUDE DEBUG] Tool use response:")
            print(response.model_dump_json(indent=2))

        return response

    def complete_with_metadata(
        self, prompt: str, system: str | None = None, cache_system: bool = False,
        response_format: dict | None = None,
    ) -> LLMResponse:
        """Generate a completion with metadata."""
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

        if system:
            if cache_system:
                kwargs["system"] = [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                kwargs["system"] = system

        response = self.client.messages.create(**kwargs)

        if os.environ.get("CLAUDE_DEBUG"):
            print("[CLAUDE DEBUG] Full response:")
            print(response.model_dump_json(indent=2))

        # Warn if the response was truncated due to token limit
        if response.stop_reason == "max_tokens":
            import sys
            print(
                f"WARNING: Claude response may be incomplete "
                f"(stop_reason=max_tokens). "
                f"Consider increasing max_tokens (current: {self.max_tokens}).",
                file=sys.stderr,
            )

        # Extract content
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        # Extract token usage
        input_tokens = getattr(response.usage, "input_tokens", 0)
        output_tokens = getattr(response.usage, "output_tokens", 0)

        # Extract cache token stats
        cache_read_tokens = getattr(response.usage, "cache_read_input_tokens", 0) or 0
        cache_creation_tokens = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
        cached = cache_read_tokens > 0

        return LLMResponse(
            content=content,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached=cached,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
        )
