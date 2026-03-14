"""OpenAI GPT backend."""

import os

from .base import LLMBackend, LLMResponse


class OpenAIBackend(LLMBackend):
    """OpenAI API backend."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 4096,
        base_url: str | None = None,
    ):
        super().__init__(model)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.max_tokens = max_tokens
        self.base_url = base_url
        self._client = None

    @property
    def default_model(self) -> str:
        return "gpt-4o"

    @property
    def client(self):
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "openai package required. Install with: pip install openai"
                ) from None

            if not self.api_key:
                raise ValueError(
                    "OpenAI API key required. Set OPENAI_API_KEY or pass api_key."
                )

            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url

            self._client = openai.OpenAI(**kwargs)
        return self._client

    def complete(
        self, prompt: str, system: str | None = None, cache_system: bool = False,
        response_format: dict | None = None,
    ) -> str:
        """Generate a completion using OpenAI."""
        response = self.complete_with_metadata(prompt, system)
        return response.content

    def complete_with_metadata(
        self, prompt: str, system: str | None = None, cache_system: bool = False,
        response_format: dict | None = None,
    ) -> LLMResponse:
        """Generate a completion with metadata."""
        messages = []

        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
        )

        content = response.choices[0].message.content or ""

        input_tokens = 0
        output_tokens = 0
        if response.usage:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

        return LLMResponse(
            content=content,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached=False,
        )
