"""
Agent loop: the master while-loop inspired by Claude Code.

Think → Act → Observe → Repeat.
Single-threaded, flat message history, terminates when model
stops producing tool calls.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from .tools import ToolRegistry, TOOL_SCHEMAS
from .index import StructuralIndex, build_structural_index
from .prompts import SYSTEM_PROMPT, format_task_prompt
from .state import ItemAssignment
from ..config import OPENROUTER_BASE_URL, LLM_TIMEOUT_SECONDS


@dataclass
class AgentResult:
    accession: str
    assignments: list[ItemAssignment]
    turns_used: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_latency_ms: int
    finalized: bool
    error: str | None = None
    event_log: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# JSON-mode fallback for models without native function calling
# ---------------------------------------------------------------------------

def _build_json_tool_prompt(tools_schemas: list[dict]) -> str:
    """Build a prompt describing available tools for JSON-mode models."""
    lines = [
        "\n## Available Tools",
        "To call a tool, respond with ONLY a JSON object in this format:",
        '{"tool": "tool_name", "args": {"arg1": "value1"}}',
        "",
        "To call multiple tools, respond with a JSON array:",
        '[{"tool": "tool_name1", "args": {}}, {"tool": "tool_name2", "args": {"arg1": "val"}}]',
        "",
        "When you are done (all items assigned and validated), respond with plain text (no JSON).",
        "",
        "Tools:",
    ]
    for schema in tools_schemas:
        fn = schema["function"]
        params = fn.get("parameters", {}).get("properties", {})
        required = fn.get("parameters", {}).get("required", [])
        param_desc = ", ".join(
            f"{k}: {v.get('type', 'any')}{' (required)' if k in required else ''}"
            for k, v in params.items()
        )
        lines.append(f"  - {fn['name']}({param_desc}): {fn['description']}")
    return "\n".join(lines)


def _parse_json_tool_calls(text: str) -> list[dict] | None:
    """Parse tool calls from JSON-mode response text."""
    text = text.strip()

    # Try to extract JSON from markdown fences
    import re
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    if not text.startswith(("{", "[")):
        return None

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Try fixing trailing commas
        cleaned = re.sub(r',\s*([}\]])', r'\1', text)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return None

    if isinstance(parsed, dict) and "tool" in parsed:
        return [parsed]
    if isinstance(parsed, list):
        return [p for p in parsed if isinstance(p, dict) and "tool" in p]
    return None


# ---------------------------------------------------------------------------
# Message compaction (prevent API overflow on large filings)
# ---------------------------------------------------------------------------

_COMPACT_CHAR_THRESHOLD = 100_000
_COMPACT_INTERVAL = 10


def _compact_messages(messages: list[dict], keep_last: int = 10) -> list[dict]:
    """Replace old tool results with summaries to reduce message size."""
    if len(messages) <= keep_last + 2:  # system + user + keep_last
        return messages
    # Keep system prompt, user prompt, and last N messages
    head = messages[:2]  # system + user
    tail = messages[-keep_last:]
    # Summarize middle section
    middle = messages[2:-keep_last]
    summary = (
        f"[Previous {len(middle)} messages compacted. "
        "The agent explored the filing structure, assigned items, "
        "and performed validation.]"
    )
    return head + [{"role": "user", "content": summary}] + tail


def _total_message_chars(messages: list[dict]) -> int:
    """Estimate total character size of the message history."""
    total = 0
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(content)
        else:
            # OpenAI message object
            content = getattr(msg, "content", None) or ""
            if isinstance(content, str):
                total += len(content)
            # Count tool call arguments
            for tc in getattr(msg, "tool_calls", None) or []:
                total += len(getattr(tc.function, "arguments", "") or "")
    return total


# ---------------------------------------------------------------------------
# Agent Loop
# ---------------------------------------------------------------------------

class AgentLoop:
    """
    The master agent loop. Model-agnostic via OpenRouter.

    Supports two modes:
    - Native function calling (OpenAI tools parameter)
    - JSON-mode fallback (tool schemas in system prompt)
    """

    def __init__(
        self,
        model_id: str,
        api_key: str,
        max_turns: int = 30,
        use_native_tools: bool = True,
    ):
        self.client = AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
        )
        self.model_id = model_id
        self.max_turns = max_turns
        self.use_native_tools = use_native_tools
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_latency_ms = 0

    async def run(
        self,
        index: StructuralIndex,
        accession: str,
    ) -> AgentResult:
        """Run the full agent loop on a filing."""
        tools = ToolRegistry(index)
        overview = tools.execute("get_filing_overview", {})

        system_prompt = SYSTEM_PROMPT
        if not self.use_native_tools:
            system_prompt += _build_json_tool_prompt(TOOL_SCHEMAS)

        task_prompt = format_task_prompt(accession, json.loads(overview))

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt},
        ]

        stall_count = 0
        prev_issue_count = float("inf")

        for turn in range(self.max_turns):
            tools.set_turn(turn)

            # --- Call model ---
            start_time = time.monotonic()
            try:
                kwargs = {
                    "model": self.model_id,
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": 4096,
                    "timeout": LLM_TIMEOUT_SECONDS,
                }
                if self.use_native_tools:
                    kwargs["tools"] = tools.openai_schemas()

                response = await self.client.chat.completions.create(**kwargs)
                elapsed_ms = int((time.monotonic() - start_time) * 1000)

            except Exception as e:
                elapsed_ms = int((time.monotonic() - start_time) * 1000)
                self.total_latency_ms += elapsed_ms
                err_str = str(e)
                is_rate_limit = "429" in err_str
                is_server_err = "5" in str(getattr(e, "status_code", ""))

                if is_rate_limit or is_server_err:
                    # Retry up to 3 times with exponential backoff
                    response = None
                    for retry in range(3):
                        wait = (retry + 1) * 10 if is_rate_limit else (retry + 1) * 3
                        await asyncio.sleep(wait)
                        try:
                            response = await self.client.chat.completions.create(**kwargs)
                            elapsed_ms = int((time.monotonic() - start_time) * 1000)
                            break
                        except Exception:
                            continue
                    if response is None:
                        return AgentResult(
                            accession=accession,
                            assignments=tools.get_state().get_assignments(),
                            turns_used=turn,
                            total_prompt_tokens=self.total_prompt_tokens,
                            total_completion_tokens=self.total_completion_tokens,
                            total_latency_ms=self.total_latency_ms,
                            finalized=False,
                            error=f"API error after 3 retries: {e}",
                            event_log=tools.get_state().get_event_log(),
                        )
                else:
                    return AgentResult(
                        accession=accession,
                        assignments=tools.get_state().get_assignments(),
                        turns_used=turn,
                        total_prompt_tokens=self.total_prompt_tokens,
                        total_completion_tokens=self.total_completion_tokens,
                        total_latency_ms=self.total_latency_ms,
                        finalized=False,
                        error=f"API error: {e}",
                        event_log=tools.get_state().get_event_log(),
                    )

            self.total_latency_ms += elapsed_ms
            usage = response.usage
            if usage:
                self.total_prompt_tokens += usage.prompt_tokens or 0
                self.total_completion_tokens += usage.completion_tokens or 0

            choice = response.choices[0]
            message = choice.message

            # --- Process response ---
            if self.use_native_tools:
                # Native function calling mode
                messages.append(message)

                if not message.tool_calls:
                    break  # Agent is done

                for tc in message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    result = tools.execute(tc.function.name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

            else:
                # JSON-mode fallback
                raw_text = message.content or ""
                messages.append({"role": "assistant", "content": raw_text})

                tool_calls = _parse_json_tool_calls(raw_text)
                if not tool_calls:
                    break  # No tool calls = agent is done

                results = []
                for tc in tool_calls:
                    tool_name = tc.get("tool", "")
                    args = tc.get("args", {})
                    result = tools.execute(tool_name, args)
                    results.append(f"[{tool_name}] {result}")

                messages.append({
                    "role": "user",
                    "content": "Tool results:\n" + "\n".join(results),
                })

            # --- Guard rails ---
            if tools.is_finalized():
                break

            # Stall detection: if validation issues aren't decreasing
            current_validation = json.loads(
                tools.execute("validate_assignments", {})
            )
            current_issues = len(current_validation.get("issues", []))
            if current_issues >= prev_issue_count and current_issues > 0:
                stall_count += 1
            else:
                stall_count = 0
            prev_issue_count = current_issues

            if stall_count >= 3:
                if turn >= 20:
                    # After 20 turns + 3 stalls, force finalize
                    messages.append({
                        "role": "user",
                        "content": (
                            "STOP iterating. You have stalled after 20+ turns. "
                            "Call finalize NOW with your current best assignments. "
                            "Further changes are making things worse, not better."
                        ),
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your last 3 changes didn't improve validation. "
                            "Try a different approach: use scan_for_heading to find "
                            "the actual heading text, or move on to other issues."
                        ),
                    })
                stall_count = 0

            # Message compaction to prevent API overflow
            if (turn + 1) % _COMPACT_INTERVAL == 0:
                msg_size = _total_message_chars(messages)
                if msg_size > _COMPACT_CHAR_THRESHOLD:
                    messages = _compact_messages(messages)

        return AgentResult(
            accession=accession,
            assignments=tools.get_state().get_assignments(),
            turns_used=turn + 1 if 'turn' in dir() else 0,
            total_prompt_tokens=self.total_prompt_tokens,
            total_completion_tokens=self.total_completion_tokens,
            total_latency_ms=self.total_latency_ms,
            finalized=tools.is_finalized(),
            event_log=tools.get_state().get_event_log(),
        )
