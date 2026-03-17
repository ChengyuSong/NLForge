"""llama.cpp local model backend."""

import json
import urllib.error
import urllib.request
from typing import Any

from .base import LLMBackend, LLMResponse


class LlamaCppBackend(LLMBackend):
    """llama.cpp local model backend using the /completion endpoint."""

    def __init__(
        self,
        model: str | None = None,
        host: str = "localhost",
        port: int = 8080,
        enable_thinking: bool = True,
    ):
        super().__init__(model)
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.enable_thinking = enable_thinking

    @property
    def default_model(self) -> str:
        # llama.cpp doesn't use model names in the same way - it serves one model at a time
        return "llama.cpp"

    @staticmethod
    def _convert_anthropic_tools_to_openai(tools: list[dict]) -> list[dict]:
        """
        Convert Anthropic tool format to OpenAI tool format.

        Anthropic format:
        {
          "name": "tool_name",
          "description": "...",
          "input_schema": {"type": "object", "properties": {...}, "required": [...]}
        }

        OpenAI format:
        {
          "type": "function",
          "function": {
            "name": "tool_name",
            "description": "...",
            "parameters": {"type": "object", "properties": {...}, "required": [...]}
          }
        }
        """
        openai_tools = []
        for tool in tools:
            # Check if already in OpenAI format
            if "type" in tool and tool["type"] == "function":
                openai_tools.append(tool)
            # Convert from Anthropic format
            elif "name" in tool and "input_schema" in tool:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool["input_schema"]
                    }
                })
            else:
                # Unknown format, pass through
                openai_tools.append(tool)
        return openai_tools

    def complete(
        self, prompt: str, system: str | None = None, cache_system: bool = False,
        response_format: dict | None = None,
    ) -> str:
        """Generate a completion using llama.cpp."""
        response = self.complete_with_metadata(prompt, system,
                                               response_format=response_format)
        return response.content

    def complete_with_metadata(
        self, prompt: str, system: str | None = None, cache_system: bool = False,
        response_format: dict | None = None,
    ) -> LLMResponse:
        """Generate a completion with metadata."""
        # Use OpenAI-compatible chat completions endpoint for better control
        url = f"{self.base_url}/v1/chat/completions"

        # Build messages array
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model if self.model != "llama.cpp" else "default",
            "messages": messages,
            "temperature": 0.5,
            # "max_tokens": 8192,
            "stream": False,
            "cache_prompt": True,
        }

        if response_format:
            payload["response_format"] = response_format

        # Add thinking mode control for Nemotron models
        if not self.enable_thinking:
            payload["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            }

        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(
                f"llama.cpp at {self.base_url} returned HTTP {e.code}: {body}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to connect to llama.cpp at {self.base_url}: {e}") from e

        # Extract content from OpenAI-compatible response format
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]
        else:
            content = ""

        # Extract token usage
        usage = result.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        return LLMResponse(
            content=content,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached=False,
        )

    @staticmethod
    def _convert_anthropic_messages_to_openai(messages: list[dict]) -> list[dict]:
        """
        Convert Anthropic message format to OpenAI message format.

        Anthropic assistant message with tool use:
        {
          "role": "assistant",
          "content": [
            {"type": "text", "text": "..."},
            {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
          ]
        }

        OpenAI assistant message with tool calls:
        {
          "role": "assistant",
          "content": "...",
          "tool_calls": [{
            "id": "...",
            "type": "function",
            "function": {"name": "...", "arguments": "{...}"}
          }]
        }

        Anthropic tool results:
        {
          "role": "user",
          "content": [{"type": "tool_result", "tool_use_id": "...", "content": "..."}]
        }

        OpenAI tool results:
        {
          "role": "tool",
          "tool_call_id": "...",
          "content": "..."
        }
        """
        openai_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            # Handle tool results in Anthropic format (user role with tool_result content)
            if role == "user" and isinstance(content, list):
                user_text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        # Convert to OpenAI tool message
                        openai_messages.append({
                            "role": "tool",
                            "tool_call_id": item.get("tool_use_id"),
                            "content": item.get("content", "")
                        })
                    elif isinstance(item, dict) and item.get("type") == "text":
                        user_text_parts.append(item.get("text", ""))
                    else:
                        # Unknown content type, pass through as user message
                        openai_messages.append(msg)
                        break
                if user_text_parts:
                    openai_messages.append({
                        "role": "user",
                        "content": "\n".join(user_text_parts),
                    })
            # Handle assistant messages with tool uses in Anthropic format
            elif role == "assistant" and isinstance(content, list):
                # Build OpenAI assistant message format
                text_content = ""
                tool_calls = []

                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_content += item.get("text", "")
                        elif item.get("type") == "tool_use":
                            # Convert Anthropic tool_use to OpenAI tool_call
                            tool_calls.append({
                                "id": item.get("id"),
                                "type": "function",
                                "function": {
                                    "name": item.get("name"),
                                    "arguments": json.dumps(item.get("input", {}))
                                }
                            })

                openai_msg: dict[str, Any] = {"role": "assistant"}

                # OpenAI requires either content or tool_calls
                if text_content:
                    openai_msg["content"] = text_content
                if tool_calls:
                    openai_msg["tool_calls"] = tool_calls

                # If neither, add empty content to satisfy OpenAI requirements
                if not text_content and not tool_calls:
                    openai_msg["content"] = ""

                openai_messages.append(openai_msg)
            else:
                # Regular message, pass through
                openai_messages.append(msg)

        return openai_messages

    def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | None = None,
    ):
        """
        Generate a completion with tool use support (OpenAI format).

        Args:
            messages: List of message dicts with role and content
            tools: Optional list of tool definitions in OpenAI or Anthropic format
            system: Optional system message

        Returns:
            Response object with tool_calls if applicable
        """
        url = f"{self.base_url}/v1/chat/completions"

        # Convert messages from Anthropic format to OpenAI format if needed
        converted_messages = self._convert_anthropic_messages_to_openai(messages)

        # Build messages array with system message if provided
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(converted_messages)

        payload = {
            "model": self.model if self.model != "llama.cpp" else "default",
            "messages": full_messages,
            "temperature": 0.6,
            # "max_tokens": 8192,
            "stream": False,
        }

        # Add thinking mode control for Nemotron models
        if not self.enable_thinking:
            payload["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            }

        if tools:
            # Convert tools from Anthropic format to OpenAI format if needed
            payload["tools"] = self._convert_anthropic_tools_to_openai(tools)

        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(
                f"llama.cpp at {self.base_url} returned HTTP {e.code}: {body}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to connect to llama.cpp at {self.base_url}: {e}") from e

        # Return a simple object that mimics the response structure
        class ToolResponse:
            def __init__(self, data):
                self.data = data
                if "choices" in data and len(data["choices"]) > 0:
                    choice = data["choices"][0]
                    self.stop_reason = choice.get("finish_reason", "end_turn")
                    self.content: list[Any] = []

                    message = choice.get("message", {})

                    # Add text content if present
                    if "content" in message and message["content"]:
                        class TextBlock:
                            def __init__(self, text):
                                self.type = "text"
                                self.text = text
                        self.content.append(TextBlock(message["content"]))

                    # Add tool calls if present
                    if "tool_calls" in message and message["tool_calls"]:
                        self.stop_reason = "tool_use"
                        class ToolUseBlock:
                            def __init__(self, tool_call):
                                self.type = "tool_use"
                                self.id = tool_call["id"]
                                self.name = tool_call["function"]["name"]
                                self.input = json.loads(tool_call["function"]["arguments"])

                        for tc in message["tool_calls"]:
                            self.content.append(ToolUseBlock(tc))
                else:
                    self.stop_reason = "end_turn"
                    self.content = []

        return ToolResponse(result)

    def is_available(self) -> bool:
        """Check if llama.cpp server is running and accessible."""
        try:
            url = f"{self.base_url}/health"
            with urllib.request.urlopen(url, timeout=10) as response:
                return bool(response.status == 200)
        except Exception:
            return False

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(model={self.model!r}, "
            f"host={self.host!r}, port={self.port})"
        )
