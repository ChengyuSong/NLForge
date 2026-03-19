"""JSON parsing utilities for LLM responses."""

import json
import re
from typing import cast

_VALID_JSON_ESCAPES = frozenset('"\\/bfnrtu')


def _fix_invalid_escapes(s: str) -> str:
    r"""Double-escape invalid ``\X`` sequences inside a JSON string literal.

    Valid JSON escapes (``\"``, ``\\``, ``\/``, ``\b``, ``\f``, ``\n``,
    ``\r``, ``\t``, ``\uXXXX``) are left alone.  Anything else
    (e.g. ``\p``, ``\s``, ``\w``) becomes ``\\p``, ``\\s``, ``\\w``.
    """
    out: list[str] = []
    i = 0
    while i < len(s):
        if s[i] == '\\' and i + 1 < len(s):
            nxt = s[i + 1]
            if nxt in _VALID_JSON_ESCAPES:
                out.append(s[i:i + 2])
                i += 2
            else:
                # Invalid escape → double the backslash
                out.append('\\\\')
                i += 1
            continue
        out.append(s[i])
        i += 1
    return "".join(out)


def repair_json(text: str) -> str:
    """Fix common JSON syntax errors from quantized LLMs.

    Handles:
      - Invalid backslash escapes in string values (``\\p`` → ``\\\\p``)
      - Trailing commas before ``}`` or ``]``
      - Mismatched closing braces/brackets (e.g. ``}`` where ``]`` expected)
      - Unescaped ``operator""X`` C++ literal operator names
    """
    # C++ user-defined literal operators: operator""h → operator\"\"h
    text = re.sub(r'operator""', r'operator\\"\\"', text)

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
            # Extract string literal (handle backslash escapes)
            j = i + 1
            while j < len(text):
                if text[j] == '\\':
                    j += 2
                elif text[j] == '"':
                    j += 1
                    break
                else:
                    j += 1
            # Fix invalid \escapes inside the string content
            raw_str = text[i:j]
            if len(raw_str) >= 2:
                inner = raw_str[1:-1] if raw_str.endswith('"') else raw_str[1:]
                fixed = _fix_invalid_escapes(inner)
                raw_str = '"' + fixed + ('"' if raw_str.endswith('"') else '')
            out.append(raw_str)
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
                return cast(dict, json.loads(json_str))
            except json.JSONDecodeError:
                try:
                    return cast(dict, json.loads(repair_json(json_str)))
                except json.JSONDecodeError as e:
                    last_err = e
        raise last_err  # type: ignore[misc]

    # Fallback: raw {…}
    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if not json_match:
        raise ValueError("No JSON found in LLM response")
    raw = json_match.group(0)
    try:
        return cast(dict, json.loads(raw))
    except json.JSONDecodeError:
        return cast(dict, json.loads(repair_json(raw)))


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
        return cast(dict, json.loads(json_str))
    except json.JSONDecodeError as e:
        if verbose:
            print(f"[ERROR] Failed to parse LLM response as JSON: {e}")
            print(f"[ERROR] Response: {text[:500]}...")
        return default_response
