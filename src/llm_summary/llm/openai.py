"""OpenAI-compatible backend (GPT, vLLM, etc.)."""

import os
import sys
from typing import Any

from .base import LLMBackend, LLMResponse


class OpenAIBackend(LLMBackend):
    """OpenAI API backend (also works with vLLM and other compatible servers)."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 16384,
        base_url: str | None = None,
        enable_thinking: bool = True,
    ):
        super().__init__(model)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.enable_thinking = enable_thinking
        self._client: Any = None

    @property
    def default_model(self) -> str:
        return "gpt-4o"

    @property
    def _uses_responses_api(self) -> bool:
        """Codex models are exposed only via the Responses API."""
        return "codex" in (self.model or "").lower()

    @property
    def _max_tokens_param(self) -> str:
        """gpt-5 and newer models require ``max_completion_tokens``."""
        name = (self.model or "").lower()
        if name.startswith("gpt-5") or name.startswith("o1") or name.startswith("o3"):
            return "max_completion_tokens"
        return "max_tokens"

    @property
    def _is_openai_endpoint(self) -> bool:
        """True when base_url is unset or points at api.openai.com."""
        if not self.base_url:
            return True
        return "api.openai.com" in self.base_url

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

            kwargs: dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url

            self._client = openai.OpenAI(**kwargs)
        return self._client

    def complete(
        self, prompt: str, system: str | None = None, cache_system: bool = False,
        response_format: dict | None = None,
    ) -> str:
        """Generate a completion using OpenAI."""
        response = self.complete_with_metadata(
            prompt, system, response_format=response_format,
        )
        return response.content

    def complete_with_metadata(
        self, prompt: str, system: str | None = None, cache_system: bool = False,
        response_format: dict | None = None,
    ) -> LLMResponse:
        """Generate a completion with metadata."""
        if self._uses_responses_api:
            return self._complete_responses(prompt, system, response_format)

        messages: list[dict[str, str]] = []

        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            self._max_tokens_param: self.max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format
        if not self.enable_thinking:
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }

        response = self.client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        content = choice.message.content or ""

        if choice.finish_reason == "length":
            print(
                f"WARNING: OpenAI response may be incomplete "
                f"(finish_reason=length). "
                f"Consider increasing max_tokens (current: {self.max_tokens}).",
                file=sys.stderr,
            )

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

    def _complete_responses(
        self,
        prompt: str,
        system: str | None,
        response_format: dict | None,
    ) -> LLMResponse:
        """Single-turn completion via the Responses API (for codex models)."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": self.max_tokens,
        }
        if system:
            kwargs["instructions"] = system
        if response_format:
            kwargs["text"] = {
                "format": _to_responses_text_format(
                    response_format, strict_schema=not self._is_openai_endpoint,
                )
            }

        response = self.client.responses.create(**kwargs)

        content = getattr(response, "output_text", "") or ""

        if getattr(response, "status", None) == "incomplete":
            reason = getattr(response, "incomplete_details", None)
            print(
                f"WARNING: OpenAI Responses output may be incomplete "
                f"(reason={reason}). Consider increasing max_tokens "
                f"(current: {self.max_tokens}).",
                file=sys.stderr,
            )

        input_tokens = 0
        output_tokens = 0
        if response.usage:
            input_tokens = getattr(response.usage, "input_tokens", 0) or 0
            output_tokens = getattr(response.usage, "output_tokens", 0) or 0

        return LLMResponse(
            content=content,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached=False,
        )

    def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | None = None,
    ) -> Any:
        """Generate a completion with tool use support.

        Accepts Anthropic-format tool definitions and messages, converts to
        OpenAI format, and returns an adapter that mimics the Anthropic
        response structure.
        """
        if self._uses_responses_api:
            return self._complete_with_tools_responses(messages, tools, system)

        oai_messages: list[dict[str, Any]] = []

        if system:
            oai_messages.append({"role": "system", "content": system})

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if isinstance(content, str):
                oai_messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                if role == "assistant":
                    # May contain text and tool_use blocks
                    text_parts = []
                    tool_calls = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                text_parts.append(item.get("text", ""))
                            elif item.get("type") == "tool_use":
                                import json
                                tool_calls.append({
                                    "id": item["id"],
                                    "type": "function",
                                    "function": {
                                        "name": item["name"],
                                        "arguments": json.dumps(item.get("input", {})),
                                    },
                                })
                    oai_msg: dict[str, Any] = {
                        "role": "assistant",
                        "content": "\n".join(text_parts) if text_parts else None,
                    }
                    if tool_calls:
                        oai_msg["tool_calls"] = tool_calls
                    oai_messages.append(oai_msg)
                elif role == "user":
                    # May contain tool_result blocks
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "tool_result":
                            oai_messages.append({
                                "role": "tool",
                                "tool_call_id": item["tool_use_id"],
                                "content": item.get("content", ""),
                            })
                        elif isinstance(item, dict) and item.get("type") == "text":
                            oai_messages.append({"role": "user", "content": item["text"]})
                        elif isinstance(item, str):
                            oai_messages.append({"role": "user", "content": item})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
            self._max_tokens_param: self.max_tokens,
        }

        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", {}),
                    },
                }
                for t in tools
            ]
        if not self.enable_thinking:
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }

        response = self.client.chat.completions.create(**kwargs)

        if os.environ.get("OPENAI_DEBUG"):
            print("[OPENAI DEBUG] Tool use response:")
            print(response.model_dump_json(indent=2))

        return _OpenAIToolResponse(response)

    def _complete_with_tools_responses(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        system: str | None,
    ) -> Any:
        """Tool-use via the Responses API (codex models)."""
        import json

        input_items: list[dict[str, Any]] = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if isinstance(content, str):
                input_items.append({"role": role, "content": content})
                continue

            if not isinstance(content, list):
                continue

            if role == "assistant":
                text_parts = []
                tool_call_items = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == "tool_use":
                        tool_call_items.append({
                            "type": "function_call",
                            "call_id": item["id"],
                            "name": item["name"],
                            "arguments": json.dumps(item.get("input", {})),
                        })
                if text_parts:
                    input_items.append({
                        "role": "assistant",
                        "content": "\n".join(text_parts),
                    })
                input_items.extend(tool_call_items)
            elif role == "user":
                for item in content:
                    if isinstance(item, str):
                        input_items.append({"role": "user", "content": item})
                    elif isinstance(item, dict):
                        if item.get("type") == "tool_result":
                            result = item.get("content", "")
                            if not isinstance(result, str):
                                result = json.dumps(result)
                            input_items.append({
                                "type": "function_call_output",
                                "call_id": item["tool_use_id"],
                                "output": result,
                            })
                        elif item.get("type") == "text":
                            input_items.append({
                                "role": "user",
                                "content": item["text"],
                            })

        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "max_output_tokens": self.max_tokens,
        }
        if system:
            kwargs["instructions"] = system
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                }
                for t in tools
            ]

        response = self.client.responses.create(**kwargs)

        if os.environ.get("OPENAI_DEBUG"):
            print("[OPENAI DEBUG] Responses tool use:")
            print(response.model_dump_json(indent=2))

        return _ResponsesToolResponse(response)


def _to_responses_text_format(
    response_format: dict, strict_schema: bool = True,
) -> dict:
    """Translate chat-completions response_format to Responses API text.format.

    When ``strict_schema`` is False (real OpenAI endpoint), downgrade
    ``json_schema`` to ``json_object`` — the JSON shape is already described in
    the prompt, and OpenAI's json_schema validator rejects schemas that don't
    meet its strict-mode constraints (every property required, no extras).
    When True (local OpenAI-compatible servers), keep the schema but inject
    ``additionalProperties: false`` recursively.
    """
    if response_format.get("type") != "json_schema":
        return response_format

    if not strict_schema:
        return {"type": "json_object"}

    schema = response_format.get("json_schema", {})
    out: dict[str, Any] = {"type": "json_schema"}
    if "name" in schema:
        out["name"] = schema["name"]
    if "schema" in schema:
        out["schema"] = _enforce_no_additional_props(schema["schema"])
    if "strict" in schema:
        out["strict"] = schema["strict"]
    return out


def _enforce_no_additional_props(schema: Any) -> Any:
    """Normalize a schema for the Responses API.

    Recursively sets ``additionalProperties: false`` and forces ``required``
    to include every key in ``properties`` (the Responses API mandates this
    for ``json_schema`` format).
    """
    if isinstance(schema, dict):
        out = {k: _enforce_no_additional_props(v) for k, v in schema.items()}
        if out.get("type") == "object":
            if "additionalProperties" not in out:
                out["additionalProperties"] = False
            props = out.get("properties")
            if isinstance(props, dict) and props:
                out["required"] = list(props.keys())
            elif "required" in out:
                del out["required"]
        return out
    if isinstance(schema, list):
        return [_enforce_no_additional_props(v) for v in schema]
    return schema


class _OpenAIToolResponse:
    """Adapter to make OpenAI responses look like Anthropic responses."""

    def __init__(self, response: Any):
        self._response = response
        self.content: list[Any] = []
        self.stop_reason = "end_turn"

        if not response.choices:
            return

        choice = response.choices[0]
        message = choice.message

        if message.content:
            self.content.append(_TextBlock(text=message.content))

        if message.tool_calls:
            import json
            self.stop_reason = "tool_use"
            for tc in message.tool_calls:
                args = tc.function.arguments
                self.content.append(
                    _ToolUseBlock(
                        id=tc.id,
                        name=tc.function.name,
                        input=json.loads(args) if args else {},
                    )
                )


class _TextBlock:
    """Mimics Anthropic TextBlock."""

    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _ToolUseBlock:
    """Mimics Anthropic ToolUseBlock."""

    def __init__(self, id: str, name: str, input: dict):
        self.type = "tool_use"
        self.id = id
        self.name = name
        self.input = input


class _ResponsesToolResponse:
    """Adapter for Responses API output mimicking the Anthropic response shape."""

    def __init__(self, response: Any):
        import json

        self._response = response
        self.content: list[Any] = []
        self.stop_reason = "end_turn"

        for item in getattr(response, "output", []) or []:
            item_type = getattr(item, "type", None)
            if item_type == "message":
                for block in getattr(item, "content", []) or []:
                    block_type = getattr(block, "type", None)
                    if block_type in ("output_text", "text"):
                        text = getattr(block, "text", "") or ""
                        if text:
                            self.content.append(_TextBlock(text=text))
            elif item_type == "function_call":
                self.stop_reason = "tool_use"
                args = getattr(item, "arguments", "") or ""
                self.content.append(
                    _ToolUseBlock(
                        id=getattr(item, "call_id", "") or getattr(item, "id", ""),
                        name=getattr(item, "name", ""),
                        input=json.loads(args) if args else {},
                    )
                )
