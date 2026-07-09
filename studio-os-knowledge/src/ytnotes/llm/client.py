"""Anthropic client wrapper for structured note extraction.

Uses **forced tool use** to obtain schema-valid JSON. This is the most portable
mechanism across ``anthropic`` SDK versions (equivalent in effect to structured
outputs for a pure extraction task): we declare a single tool whose input_schema
is the enum-constrained note schema and force the model to call it, then read the
validated ``tool_use.input`` back as our object.

Cost controls:
- The large static prefix (system prompt + serialized taxonomy) is marked with
  ``cache_control: ephemeral`` so it is reused across videos in a back-to-back
  run. All per-video content goes in the user turn so the prefix stays stable.
- ``temperature`` / ``top_p`` are intentionally NOT passed (rejected on Opus 4.8
  / Sonnet 5); steer via the prompt.
"""

from __future__ import annotations

import json

import anthropic

TOOL_NAME = "emit_note"
MAX_TOKENS = 2000


class LLMError(RuntimeError):
    pass


def make_client(api_key: str) -> anthropic.Anthropic:
    # The SDK auto-retries 429/5xx (default 2); bump slightly for CI resilience.
    return anthropic.Anthropic(api_key=api_key, max_retries=4)


def call_structured(
    client: anthropic.Anthropic,
    *,
    model: str,
    system_prompt: str,
    note_schema: dict,
    user_content: str,
    use_thinking: bool = False,
) -> dict:
    """Force a single ``emit_note`` tool call and return its validated input dict."""
    tools = [
        {
            "name": TOOL_NAME,
            "description": "Emit the structured knowledge note for this transcript.",
            "input_schema": note_schema,
        }
    ]

    kwargs: dict = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "tools": tools,
        "tool_choice": {"type": "tool", "name": TOOL_NAME},
        "messages": [{"role": "user", "content": user_content}],
    }
    if use_thinking:
        # Thinking is incompatible with forced tool_choice; fall back to auto.
        kwargs["tool_choice"] = {"type": "auto"}
        kwargs["thinking"] = {"type": "adaptive"}

    try:
        resp = client.messages.create(**kwargs)
    except anthropic.APIError as exc:
        raise LLMError(f"Anthropic API error: {exc}") from exc

    _log_cache_usage(resp)

    for block in resp.content:
        if getattr(block, "type", None) == "tool_use" and block.name == TOOL_NAME:
            data = block.input
            if isinstance(data, str):  # defensive: some paths return a JSON string
                data = json.loads(data)
            return data

    raise LLMError("model did not return an emit_note tool call")


def _log_cache_usage(resp) -> None:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return
    read = getattr(usage, "cache_read_input_tokens", 0) or 0
    write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    print(
        f"[llm] tokens in={getattr(usage, 'input_tokens', '?')} "
        f"out={getattr(usage, 'output_tokens', '?')} "
        f"cache_read={read} cache_write={write}"
    )
