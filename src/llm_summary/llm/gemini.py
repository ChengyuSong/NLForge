"""Google Gemini backend via Vertex AI."""

import base64
import os
from typing import Any

from .base import LLMBackend, LLMResponse


def _sig_to_str(sig: Any) -> str | None:
    """Convert thought_signature (bytes or str) to str for JSON safety."""
    if sig is None:
        return None
    if isinstance(sig, bytes):
        return base64.b64encode(sig).decode("ascii")
    return str(sig)


def _sig_to_bytes(sig: Any) -> bytes | None:
    """Convert thought_signature str back to bytes for the Gemini API."""
    if sig is None:
        return None
    if isinstance(sig, str):
        return base64.b64decode(sig)
    return bytes(sig)


class GeminiBackend(LLMBackend):
    """Google Gemini backend using the google-genai SDK."""

    def __init__(
        self,
        model: str | None = None,
        project_id: str | None = None,
        location: str | None = None,
        max_tokens: int = 32768,
    ):
        super().__init__(model)
        self.project_id = (
            project_id
            or os.environ.get("VERTEX_AI_PROJECT")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("CLOUD_ML_PROJECT_ID")
        )
        self.location = location or os.environ.get("VERTEX_AI_LOCATION", "global")
        self.max_tokens = max_tokens
        self._client = None

    @property
    def default_model(self) -> str:
        return "gemini-2.5-flash"

    @property
    def client(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError:
                raise ImportError(
                    "google-genai package required. Install with: pip install google-genai"
                ) from None

            if not self.project_id:
                raise ValueError(
                    "GCP project ID required. Set GOOGLE_CLOUD_PROJECT or pass project_id."
                )

            self._client = genai.Client(
                vertexai=True,
                project=self.project_id,
                location=self.location,
            )
        return self._client

    def complete(
        self, prompt: str, system: str | None = None, cache_system: bool = False,
        response_format: dict | None = None,
    ) -> str:
        """Generate a completion using Gemini."""
        response = self.complete_with_metadata(prompt, system)
        return response.content

    def complete_with_metadata(
        self, prompt: str, system: str | None = None, cache_system: bool = False,
        response_format: dict | None = None,
    ) -> LLMResponse:
        """Generate a completion with metadata."""
        from google.genai import types

        config = types.GenerateContentConfig(
            max_output_tokens=self.max_tokens,
        )
        if system:
            config.system_instruction = system

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )

        content = response.text or ""

        # Warn if the response was truncated due to token limit
        if response.candidates:
            candidate = response.candidates[0]
            finish_reason = getattr(candidate, "finish_reason", None)
            # FinishReason.STOP == 1; anything else (MAX_TOKENS==2, etc.) is abnormal
            if finish_reason is not None and str(finish_reason) not in (
                "FinishReason.STOP", "1", "STOP",
            ):
                import sys
                print(
                    f"WARNING: Gemini response may be incomplete "
                    f"(finish_reason={finish_reason}). "
                    f"Consider increasing max_tokens (current: {self.max_tokens}).",
                    file=sys.stderr,
                )

        # Extract token usage
        input_tokens = 0
        output_tokens = 0
        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0

        return LLMResponse(
            content=content,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached=False,
        )

    @staticmethod
    def _convert_tools(tools: list[dict]) -> list:
        """Convert Anthropic tool definitions to Gemini format.

        Anthropic: {"name": "...", "description": "...", "input_schema": {...}}
        Gemini: types.Tool(function_declarations=[FunctionDeclaration(...)])
        """
        from google.genai import types

        declarations = []
        for tool in tools:
            declarations.append(
                types.FunctionDeclaration(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    parameters=tool.get("input_schema"),
                )
            )
        return [types.Tool(function_declarations=declarations)]

    @staticmethod
    def _convert_messages(messages: list[dict]) -> tuple[list, dict[str, str]]:
        """Convert Anthropic-format messages to Gemini Content objects.

        Returns (contents, id_to_name_map) where id_to_name_map tracks
        tool_use id -> function name for use in tool_result messages.
        """
        from google.genai import types

        contents = []
        # Track tool_use id -> name mapping across all messages
        id_to_name: dict[str, str] = {}

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user" and isinstance(content, str):
                contents.append(
                    types.Content(role="user", parts=[types.Part(text=content)])
                )

            elif role == "user" and isinstance(content, list):
                # Could be tool_result blocks
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        tool_use_id = item["tool_use_id"]
                        func_name = id_to_name.get(tool_use_id, "unknown")
                        result_content = item.get("content", "")
                        parts.append(
                            types.Part(
                                function_response=types.FunctionResponse(
                                    id=tool_use_id,
                                    name=func_name,
                                    response={"result": result_content},
                                )
                            )
                        )
                    elif isinstance(item, dict) and item.get("type") == "text":
                        parts.append(types.Part(text=item["text"]))
                    elif isinstance(item, str):
                        parts.append(types.Part(text=item))
                if parts:
                    contents.append(types.Content(role="user", parts=parts))

            elif role == "assistant" and isinstance(content, str):
                contents.append(
                    types.Content(role="model", parts=[types.Part(text=content)])
                )

            elif role == "assistant" and isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text = item.get("text", "")
                            if text:
                                part_kwargs = {"text": text}
                                if item.get("thought"):
                                    part_kwargs["thought"] = True
                                if item.get("thought_signature"):
                                    part_kwargs["thought_signature"] = (
                                        _sig_to_bytes(item["thought_signature"])
                                    )
                                parts.append(types.Part(**part_kwargs))
                        elif item.get("type") == "tool_use":
                            tool_id = item["id"]
                            func_name = item["name"]
                            id_to_name[tool_id] = func_name
                            part_kwargs = {
                                "function_call": types.FunctionCall(
                                    id=tool_id,
                                    name=func_name,
                                    args=item.get("input", {}),
                                ),
                            }
                            if item.get("thought_signature"):
                                part_kwargs["thought_signature"] = (
                                    _sig_to_bytes(item["thought_signature"])
                                )
                            parts.append(types.Part(**part_kwargs))
                if parts:
                    contents.append(types.Content(role="model", parts=parts))

        return contents, id_to_name

    def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | None = None,
    ) -> Any:
        """Generate a completion with tool use support.

        Args:
            messages: List of message dicts in Anthropic format
            tools: Optional list of tool definitions in Anthropic format
            system: Optional system message

        Returns:
            Response adapter mimicking Anthropic response structure
        """
        from google.genai import types

        config = types.GenerateContentConfig(
            max_output_tokens=self.max_tokens,
        )

        if system:
            config.system_instruction = system

        if tools:
            config.tools = self._convert_tools(tools)

        contents, id_to_name = self._convert_messages(messages)

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        if os.environ.get("GEMINI_DEBUG"):
            print("[GEMINI DEBUG] Response:")
            print(response)

        return _GeminiToolResponse(response)


class _GeminiToolResponse:
    """Adapter to make Gemini responses look like Anthropic responses."""

    def __init__(self, response):
        self._response = response
        self.content = []
        self.stop_reason = "end_turn"

        if not response.candidates:
            return

        candidate = response.candidates[0]
        has_function_call = False

        for part in candidate.content.parts:
            if part.function_call:
                has_function_call = True
                self.content.append(
                    _ToolUseBlock(
                        id=part.function_call.id or _generate_tool_id(),
                        name=part.function_call.name,
                        input=dict(part.function_call.args) if part.function_call.args else {},
                        thought_signature=_sig_to_str(getattr(part, "thought_signature", None)),
                    )
                )
            elif part.text:
                self.content.append(
                    _TextBlock(
                        text=part.text,
                        thought=getattr(part, "thought", False),
                        thought_signature=_sig_to_str(getattr(part, "thought_signature", None)),
                    )
                )

        if has_function_call:
            self.stop_reason = "tool_use"


class _TextBlock:
    """Mimics Anthropic TextBlock."""

    def __init__(self, text: str, thought: bool = False, thought_signature: str | None = None):
        self.type = "text"
        self.text = text
        self.thought = thought
        self.thought_signature = thought_signature


class _ToolUseBlock:
    """Mimics Anthropic ToolUseBlock."""

    def __init__(self, id: str, name: str, input: dict, thought_signature: str | None = None):
        self.type = "tool_use"
        self.id = id
        self.name = name
        self.input = input
        self.thought_signature = thought_signature


_tool_id_counter = 0


def _generate_tool_id() -> str:
    """Generate a unique tool use ID when Gemini doesn't provide one."""
    global _tool_id_counter
    _tool_id_counter += 1
    return f"toolu_gemini_{_tool_id_counter:04d}"
