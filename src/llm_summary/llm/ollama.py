"""Ollama local model backend."""

import json
import urllib.error
import urllib.request

from .base import LLMBackend, LLMResponse


class OllamaBackend(LLMBackend):
    """Ollama local model backend."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str = "http://localhost:11434",
    ):
        super().__init__(model)
        self.base_url = base_url.rstrip("/")

    @property
    def default_model(self) -> str:
        return "llama3.1"

    def complete(
        self, prompt: str, system: str | None = None, cache_system: bool = False,
        response_format: dict | None = None,
    ) -> str:
        """Generate a completion using Ollama."""
        response = self.complete_with_metadata(prompt, system)
        return response.content

    def complete_with_metadata(
        self, prompt: str, system: str | None = None, cache_system: bool = False,
        response_format: dict | None = None,
    ) -> LLMResponse:
        """Generate a completion with metadata."""
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        if system:
            payload["system"] = system

        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to connect to Ollama at {self.base_url}: {e}") from e

        content = result.get("response", "")

        # Ollama doesn't always provide token counts
        input_tokens = result.get("prompt_eval_count", 0)
        output_tokens = result.get("eval_count", 0)

        return LLMResponse(
            content=content,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached=False,
        )

    def list_models(self) -> list[str]:
        """List available models in Ollama."""
        url = f"{self.base_url}/api/tags"

        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to connect to Ollama at {self.base_url}: {e}") from e

        models = result.get("models", [])
        return [m.get("name", "") for m in models]

    def is_available(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            self.list_models()
            return True
        except Exception:
            return False
