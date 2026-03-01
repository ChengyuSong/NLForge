"""JSON parsing utilities for LLM responses."""

import json
import re


def repair_json(text: str) -> str:
    """Fix common JSON syntax errors from quantized LLMs.

    Handles:
      - Trailing commas before ``}`` or ``]``
      - Mismatched closing braces/brackets (e.g. ``}`` where ``]`` expected)
    """
    # Trailing commas
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # Fix mismatched closers by tracking a nesting stack.
    # Respects JSON string literals so structural chars inside strings
    # are left alone.
    out: list[str] = []
    stack: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '"':
            # Skip string literal (handle backslash escapes)
            j = i + 1
            while j < len(text):
                if text[j] == '\\':
                    j += 2
                elif text[j] == '"':
                    j += 1
                    break
                else:
                    j += 1
            out.append(text[i:j])
            i = j
            continue
        if ch in ('{', '['):
            stack.append('}' if ch == '{' else ']')
            out.append(ch)
        elif ch in ('}', ']'):
            if stack:
                out.append(stack.pop())
            else:
                out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def extract_json(response: str) -> dict:
    """Extract and parse JSON from an LLM response.

    Tries fenced ``json`` blocks in reverse order (last block is usually the
    final answer when the model dumps draft JSON during reasoning).  Falls
    back to a raw ``{…}`` search.  Applies :func:`repair_json` automatically
    when ``json.loads`` fails.  Raises on total failure.
    """
    json_blocks = re.findall(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    if json_blocks:
        last_err: Exception | None = None
        for json_str in reversed(json_blocks):
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                try:
                    return json.loads(repair_json(json_str))
                except json.JSONDecodeError as e:
                    last_err = e
        raise last_err  # type: ignore[misc]

    # Fallback: raw {…}
    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if not json_match:
        raise ValueError("No JSON found in LLM response")
    raw = json_match.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(repair_json(raw))


def strip_markdown_json(text: str) -> str:
    """
    Strip ```json``` code blocks from LLM response.

    Args:
        text: LLM response that may contain markdown code blocks

    Returns:
        Cleaned JSON string
    """
    json_str = text.strip()
    if json_str.startswith("```json"):
        json_str = json_str[7:]  # Remove ```json
    if json_str.startswith("```"):
        json_str = json_str[3:]  # Remove ```
    if json_str.endswith("```"):
        json_str = json_str[:-3:]  # Remove trailing ```
    return json_str.strip()


def parse_llm_json(
    text: str,
    default_response: dict | None = None,
    verbose: bool = False,
) -> dict:
    """
    Parse JSON from LLM response with error handling.

    Args:
        text: LLM response containing JSON
        default_response: Fallback dict to return on parse error
        verbose: If True, print error messages

    Returns:
        Parsed JSON dict or default_response on error
    """
    if default_response is None:
        default_response = {}

    try:
        json_str = strip_markdown_json(text)
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        if verbose:
            print(f"[ERROR] Failed to parse LLM response as JSON: {e}")
            print(f"[ERROR] Response: {text[:500]}...")
        return default_response
