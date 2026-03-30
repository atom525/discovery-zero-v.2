"""
LLM runtime for Discovery Zero.

Provides:
  - LiteLLM-compatible OpenAI API integration
  - httpx-backed transport with connection pool + exponential-back-off retries
  - Structured output via response_format (JSON schema constrained decoding)
  - Best-of-N sampling (n > 1)
  - Token budget tracking
  - Skill loading and JSON extraction

All callers should use the module-level convenience functions (run_skill,
chat_completion, list_models) which automatically use the shared transport
instance.  Internals that need explicit dependency injection can pass a
transport= argument.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

from discovery_zero.tools.llm_transport import LLMTransport, TransportError, get_default_transport
from discovery_zero.tools.llm_budget import TokenBudget, BudgetExhaustedError

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Constants (kept for backward-compat; prefer config.CONFIG)          #
# ------------------------------------------------------------------ #
ENV_API_BASE = "LITELLM_PROXY_API_BASE"
ENV_API_KEY = "LITELLM_PROXY_API_KEY"
ENV_MODEL = "DISCOVERY_ZERO_LLM_MODEL"
DEFAULT_MODEL = "cds/Claude-4.6-opus"
DEFAULT_TIMEOUT_SECONDS = 300
LOCAL_ENV_FILE = ".env.local"

_LOCAL_ENV_LOADED = False


# ------------------------------------------------------------------ #
# Exceptions                                                           #
# ------------------------------------------------------------------ #

class LLMError(RuntimeError):
    """Raised when the configured LLM endpoint returns an error or
    when JSON extraction from model output fails."""


# ------------------------------------------------------------------ #
# Config                                                               #
# ------------------------------------------------------------------ #

@dataclass
class LLMConfig:
    api_base: str
    api_key: str
    model: str = DEFAULT_MODEL
    timeout: int = DEFAULT_TIMEOUT_SECONDS


def get_skill_root() -> Path:
    """Resolve the Discovery Zero project root (top of the package tree)."""
    return Path(__file__).resolve().parent.parent.parent.parent


def _load_local_env_file() -> None:
    global _LOCAL_ENV_LOADED
    if _LOCAL_ENV_LOADED:
        return
    env_path = get_skill_root() / LOCAL_ENV_FILE
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)
    _LOCAL_ENV_LOADED = True


def get_llm_config(
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> LLMConfig:
    """Build LLMConfig from arguments or environment variables."""
    _load_local_env_file()
    resolved_base = api_base or os.environ.get(ENV_API_BASE, "").strip()
    resolved_key = api_key or os.environ.get(ENV_API_KEY, "").strip()
    resolved_model = (
        model
        or os.environ.get(ENV_MODEL, "").strip()
        or DEFAULT_MODEL
    )
    if not resolved_base:
        raise LLMError(f"Missing {ENV_API_BASE}.")
    if not resolved_key:
        raise LLMError(f"Missing {ENV_API_KEY}.")
    return LLMConfig(
        api_base=resolved_base.rstrip("/"),
        api_key=resolved_key,
        model=resolved_model,
        timeout=timeout,
    )


# ------------------------------------------------------------------ #
# Low-level API call                                                   #
# ------------------------------------------------------------------ #

def _auth_headers(api_key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _extract_stream_delta_text(delta: Any) -> str:
    """Extract text content from one streamed delta/message payload."""
    if isinstance(delta, str):
        return delta
    if not isinstance(delta, dict):
        return ""
    content = delta.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


class _StreamingTextRecorder:
    """Incrementally parse SSE-style chunks and append assistant text to a file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self._buffer = ""
        self._saw_sse = False

    @property
    def saw_sse(self) -> bool:
        return self._saw_sse

    def feed(self, raw_chunk: str) -> None:
        self._buffer += raw_chunk
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._consume_line(line.rstrip("\r"))

    def finalize(self) -> None:
        if self._buffer:
            self._consume_line(self._buffer.rstrip("\r"))
            self._buffer = ""

    def overwrite_final_text(self, text: str) -> None:
        self.path.write_text(text, encoding="utf-8")

    def _consume_line(self, line: str) -> None:
        stripped = line.strip()
        if not stripped or stripped.startswith(":") or not stripped.startswith("data:"):
            return
        payload = stripped[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            return
        try:
            event = json.loads(payload)
        except Exception:
            return
        self._saw_sse = True
        parts: List[str] = []
        for choice in event.get("choices", []) or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict):
                text = _extract_stream_delta_text(delta)
                if text:
                    parts.append(text)
            message = choice.get("message")
            if isinstance(message, dict):
                text = _extract_stream_delta_text(message)
                if text:
                    parts.append(text)
        if parts:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write("".join(parts))


def _aggregate_streamed_chat_response(
    chunks: List[str],
    *,
    model: str,
) -> Dict[str, Any]:
    """Aggregate SSE/text chunks into a normal chat-completion response dict.

    Supports:
    - OpenAI/LiteLLM SSE format: ``data: {...}``
    - Fallback non-stream JSON body if the upstream ignores ``stream=true``
    """
    raw_text = "".join(chunks).strip()
    if not raw_text:
        raise LLMError("Streamed LLM response was empty.")

    # Some upstreams may ignore stream=True and return one normal JSON body.
    if raw_text.startswith("{"):
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict) and "choices" in parsed:
            return parsed

    choices_by_index: Dict[int, Dict[str, Any]] = {}
    usage: Dict[str, Any] = {}
    buffer = raw_text
    parsed_any = False

    # Parse SSE lines.
    for raw_line in buffer.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        event = json.loads(payload)
        parsed_any = True
        if isinstance(event.get("usage"), dict):
            usage = event["usage"]
        for choice in event.get("choices", []) or []:
            if not isinstance(choice, dict):
                continue
            index = int(choice.get("index", 0))
            agg = choices_by_index.setdefault(
                index,
                {
                    "index": index,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                },
            )
            delta = choice.get("delta")
            if isinstance(delta, dict):
                role = delta.get("role")
                if isinstance(role, str) and role:
                    agg["message"]["role"] = role
                text = _extract_stream_delta_text(delta)
                if text:
                    agg["message"]["content"] += text
            message = choice.get("message")
            if isinstance(message, dict):
                role = message.get("role")
                if isinstance(role, str) and role:
                    agg["message"]["role"] = role
                text = _extract_stream_delta_text(message)
                if text:
                    agg["message"]["content"] += text
            finish_reason = choice.get("finish_reason")
            if finish_reason is not None:
                agg["finish_reason"] = finish_reason

    if not parsed_any:
        raise LLMError("Streamed LLM response did not contain valid SSE data.")

    choices = [choices_by_index[idx] for idx in sorted(choices_by_index)]
    if not choices:
        raise LLMError("Streamed LLM response did not contain any choices.")

    return {
        "id": f"streamed-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
        "usage": usage,
    }


def list_models(
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    transport: Optional[LLMTransport] = None,
) -> List[str]:
    """List model IDs from a LiteLLM-compatible /v1/models endpoint."""
    config = get_llm_config(api_base=api_base, api_key=api_key, model=model)
    t = transport or get_default_transport()
    url = f"{config.api_base}/v1/models"
    try:
        data = t.post_json(
            url,
            {},
            headers={**_auth_headers(config.api_key), "Accept": "application/json"},
        )
    except Exception:
        # /v1/models is a GET endpoint — fall back to urllib GET
        import urllib.request
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {config.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    return [item["id"] for item in data.get("data", []) if "id" in item]


def chat_completion(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    *,
    response_format: Optional[Dict[str, Any]] = None,
    n: int = 1,
    transport: Optional[LLMTransport] = None,
    budget: Optional[TokenBudget] = None,
    skill: str = "",
    node_id: str = "",
    stream: Optional[bool] = None,
    stream_record_path: Optional[Path] = None,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Run a chat completion against the configured LiteLLM proxy.

    Args:
        messages: OpenAI-style message list.
        model: Override model name.
        response_format: JSON schema constrained decoding spec, e.g.:
            ``{"type": "json_object"}`` or
            ``{"type": "json_schema", "json_schema": {...}}``
        n: Number of completion candidates (best-of-n).  When n > 1,
            returns a list of response dicts, one per choice.
        transport: Explicit transport (uses module default if None).
        budget: Optional TokenBudget to track usage.
        skill: Skill name for budget annotation.
        node_id: Node identifier for budget annotation.
        stream: Whether to prefer streaming responses.  When None, uses the
            global config and defaults to streaming for n=1 only.
        stream_record_path: Optional file path to record generated assistant
            text while streaming. When provided, the file is updated during
            generation and overwritten with the final assistant text on success.

    Returns:
        A single response dict when n=1, or list[dict] when n>1.
    """
    config = get_llm_config(
        api_base=api_base,
        api_key=api_key,
        model=model,
        timeout=timeout,
    )
    t = transport or get_default_transport()

    try:
        from discovery_zero.config import CONFIG as _cfg
        _max_output_tokens = _cfg.llm_max_output_tokens
        _auto_continue_limit = _cfg.llm_auto_continue
    except Exception:
        _max_output_tokens = 16000
        _auto_continue_limit = 3

    payload: Dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "n": n,
    }

    if _max_output_tokens > 0:
        payload["max_tokens"] = _max_output_tokens

    if not (config.model.startswith("gpt-5") and temperature == 0.0):
        payload["temperature"] = temperature

    if response_format is not None:
        payload["response_format"] = response_format

    if budget is not None:
        budget.check_before_call(skill=skill, node_id=node_id)

    url = f"{config.api_base}/v1/chat/completions"
    should_stream = False
    if stream is None:
        try:
            from discovery_zero.config import CONFIG
            should_stream = bool(CONFIG.llm_streaming) and n == 1
        except Exception:
            should_stream = n == 1
    else:
        should_stream = bool(stream) and n == 1

    try:
        if should_stream:
            recorder = _StreamingTextRecorder(stream_record_path) if stream_record_path else None
            stream_payload = {
                **payload,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            chunks: List[str] = []
            for chunk in t.post_stream(
                url,
                stream_payload,
                timeout=float(timeout),
                headers={**_auth_headers(config.api_key), "Accept": "text/event-stream"},
            ):
                chunks.append(chunk)
                if recorder is not None:
                    recorder.feed(chunk)
            if recorder is not None:
                recorder.finalize()
            response = _aggregate_streamed_chat_response(chunks, model=config.model)
            if recorder is not None:
                recorder.overwrite_final_text(extract_text_content(response))
        else:
            response = t.post_json(
                url,
                payload,
                timeout=float(timeout),
                headers=_auth_headers(config.api_key),
            )
            if stream_record_path is not None:
                stream_record_path.parent.mkdir(parents=True, exist_ok=True)
                stream_record_path.write_text(
                    extract_text_content(response),
                    encoding="utf-8",
                )
    except TransportError as exc:
        if should_stream:
            logger.warning("Streaming LLM request failed, falling back to non-streaming: %s", exc)
            try:
                response = t.post_json(
                    url,
                    payload,
                    timeout=float(timeout),
                    headers=_auth_headers(config.api_key),
                )
                if stream_record_path is not None:
                    stream_record_path.parent.mkdir(parents=True, exist_ok=True)
                    stream_record_path.write_text(
                        extract_text_content(response),
                        encoding="utf-8",
                    )
            except TransportError as fallback_exc:
                raise LLMError(f"LLM request failed: {fallback_exc}") from fallback_exc
        else:
            raise LLMError(f"LLM request failed: {exc}") from exc
    except LLMError as exc:
        if should_stream:
            logger.warning("Streaming parse failed, falling back to non-streaming: %s", exc)
            try:
                response = t.post_json(
                    url,
                    payload,
                    timeout=float(timeout),
                    headers=_auth_headers(config.api_key),
                )
                if stream_record_path is not None:
                    stream_record_path.parent.mkdir(parents=True, exist_ok=True)
                    stream_record_path.write_text(
                        extract_text_content(response),
                        encoding="utf-8",
                    )
            except TransportError as fallback_exc:
                raise LLMError(f"LLM request failed: {fallback_exc}") from fallback_exc
        else:
            raise

    if budget is not None:
        usage = response.get("usage", {})
        budget.record(usage, model=config.model, skill=skill, node_id=node_id)

    # --- Auto-continuation on truncation (finish_reason == "length") ---
    if n == 1 and _auto_continue_limit > 0 and response_format is None:
        accumulated_text = extract_text_content(response)
        continuation_round = 0
        while continuation_round < _auto_continue_limit:
            fr = (response.get("choices") or [{}])[0].get("finish_reason", "stop")
            if fr != "length":
                break
            continuation_round += 1
            logger.info("Output truncated (finish_reason=length), auto-continuing (%d/%d)",
                        continuation_round, _auto_continue_limit)
            cont_messages = list(messages) + [
                {"role": "assistant", "content": accumulated_text},
                {"role": "user", "content": "Continue from where you left off. Do not repeat what you already wrote."},
            ]
            cont_payload: Dict[str, Any] = {
                "model": config.model,
                "messages": cont_messages,
                "n": 1,
            }
            if _max_output_tokens > 0:
                cont_payload["max_tokens"] = _max_output_tokens
            if not (config.model.startswith("gpt-5") and temperature == 0.0):
                cont_payload["temperature"] = temperature
            try:
                response = t.post_json(
                    url, cont_payload,
                    timeout=float(timeout),
                    headers=_auth_headers(config.api_key),
                )
            except TransportError as exc:
                logger.warning("Auto-continue request failed: %s", exc)
                break
            chunk_text = extract_text_content(response)
            accumulated_text += chunk_text
            if budget is not None:
                cont_usage = response.get("usage", {})
                budget.record(cont_usage, model=config.model, skill=skill, node_id=node_id)

        if continuation_round > 0:
            response = _make_single_text_response(accumulated_text, response)
            if stream_record_path is not None:
                stream_record_path.parent.mkdir(parents=True, exist_ok=True)
                stream_record_path.write_text(accumulated_text, encoding="utf-8")

    if n == 1:
        return response

    # For best-of-n: return one response dict per choice, each with the
    # standard "choices" list containing exactly that one choice.
    choices = response.get("choices", [])
    result = []
    for choice in choices:
        single = dict(response)
        single["choices"] = [choice]
        result.append(single)
    return result


def _make_single_text_response(text: str, base_response: Dict[str, Any]) -> Dict[str, Any]:
    """Build a synthetic response dict with concatenated text and finish_reason=stop."""
    resp = dict(base_response)
    resp["choices"] = [{
        "index": 0,
        "message": {"role": "assistant", "content": text},
        "finish_reason": "stop",
    }]
    return resp


# ------------------------------------------------------------------ #
# Text / JSON extraction                                               #
# ------------------------------------------------------------------ #

def extract_text_content(response: Dict[str, Any]) -> str:
    """Extract the assistant text content from a chat completion response."""
    choices = response.get("choices", [])
    if not choices:
        raise LLMError("LLM response did not contain any choices.")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "\n".join(parts)
    return str(content)


def _repair_json_string(text: str) -> str:
    """Apply heuristic repairs to common LLM JSON formatting errors.

    Handles:
    - Trailing commas before } or ]
    - Single-quoted string keys/values (Python-style dict output)
    - Truncated JSON: tries to close unclosed braces/brackets
    """
    import re as _re

    # Remove trailing commas before closing braces/brackets
    text = _re.sub(r",\s*([}\]])", r"\1", text)

    # Replace single-quoted strings with double-quoted ones only when safe.
    # Strategy: replace 'key' and 'value' patterns not inside double-quoted strings.
    # This is a best-effort heuristic and may fail on complex inputs.
    def _single_to_double(m: "_re.Match[str]") -> str:
        inner = m.group(1)
        # Escape any internal double quotes, unescape internal single quotes
        inner = inner.replace('"', '\\"').replace("\\'", "'")
        return f'"{inner}"'

    # Only replace 'text' patterns that look like JSON string tokens
    text = _re.sub(r"(?<![\\])'([^'\\]*(?:\\.[^'\\]*)*)'", _single_to_double, text)

    # Attempt to close unclosed braces/brackets (truncated JSON)
    open_braces = text.count("{") - text.count("}")
    open_brackets = text.count("[") - text.count("]")
    if open_braces > 0 or open_brackets > 0:
        # Close in LIFO order by scanning the text
        stack: list[str] = []
        for ch in text:
            if ch == "{":
                stack.append("}")
            elif ch == "[":
                stack.append("]")
            elif ch in ("}", "]") and stack and stack[-1] == ch:
                stack.pop()
        text = text.rstrip().rstrip(",").rstrip()
        text += "".join(reversed(stack))

    return text


def extract_json_block(text: str) -> Any:
    """
    Extract the first JSON object or array from raw model output.

    Handles:
      - Output wrapped in a ```json ... ``` code fence
      - Bare JSON at the start of output
      - JSON buried inside prose (scan for first { or [)
      - Common LLM formatting errors (trailing commas, single quotes, truncation)
    """
    stripped = text.strip()

    # Strip code fence (handles ```json, ```python, ``` etc.)
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    # Try to parse the whole stripped string as-is
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Try after heuristic repairs
    try:
        repaired = _repair_json_string(stripped)
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Scan for the outermost JSON value
    start_candidates = [i for i in (stripped.find("{"), stripped.find("[")) if i != -1]
    if not start_candidates:
        raise LLMError("Model output did not contain JSON.")
    start = min(start_candidates)
    end_curly = stripped.rfind("}")
    end_square = stripped.rfind("]")
    end = max(end_curly, end_square)
    if end < start:
        raise LLMError("Model output contained malformed JSON boundaries.")
    snippet = stripped[start:end + 1]

    # Try snippet as-is
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        pass

    # Try snippet after repairs
    try:
        repaired_snippet = _repair_json_string(snippet)
        return json.loads(repaired_snippet)
    except json.JSONDecodeError as exc:
        raise LLMError(
            f"Failed to parse JSON from model output: {exc}\nRaw output:\n{text[:800]}"
        ) from exc


# ------------------------------------------------------------------ #
# Skill execution                                                      #
# ------------------------------------------------------------------ #

def load_skill_prompt(skill_filename: str) -> str:
    """Load a skill markdown file from `skills/`."""
    path = get_skill_root() / "skills" / skill_filename
    if not path.exists():
        raise FileNotFoundError(f"Skill file not found: {path}")
    return path.read_text(encoding="utf-8")


def run_skill(
    skill_filename: str,
    task_input: str,
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    *,
    response_format: Optional[Dict[str, Any]] = None,
    n: int = 1,
    transport: Optional[LLMTransport] = None,
    budget: Optional[TokenBudget] = None,
    node_id: str = "",
    schema: Optional[Dict[str, Any]] = None,
    record_path: Optional[Path] = None,
) -> Union[tuple[str, Any], List[tuple[str, Any]]]:
    """
    Run an LLM-backed skill and parse the JSON result.

    Args:
        skill_filename: Filename under skills/ (e.g. "plausible.skill.md").
        task_input: User-side prompt content.
        schema: Optional JSON Schema for constrained decoding.  When provided
            it is passed as response_format to the API (json_schema mode) with
            a fallback to json_object mode if the API doesn't support it.
        n: Best-of-n sampling.  Returns list[(raw, parsed)] when n > 1.
        response_format: Direct override of response_format (takes precedence
            over schema).

    Returns:
        ``(raw_text, parsed_json)`` for n=1, or list of such tuples for n>1.
    """
    skill_prompt = load_skill_prompt(skill_filename)

    # Build effective response_format
    eff_response_format = response_format
    if eff_response_format is None and schema is not None:
        skill_name = skill_filename.removesuffix(".skill.md").replace(".", "_")
        eff_response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": skill_name,
                "schema": schema,
                "strict": True,
            },
        }

    response = chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    skill_prompt
                    + "\n\nReturn ONLY a valid JSON response matching the required output format."
                ),
            },
            {"role": "user", "content": task_input},
        ],
        model=model,
        api_base=api_base,
        api_key=api_key,
        temperature=temperature,
        timeout=timeout,
        response_format=eff_response_format,
        n=n,
        transport=transport,
        budget=budget,
        skill=skill_filename,
        node_id=node_id,
        stream_record_path=record_path,
    )

    if n == 1:
        assert isinstance(response, dict)
        raw = extract_text_content(response)
        try:
            parsed = extract_json_block(raw)
        except LLMError:
            # If constrained decoding was requested but JSON still failed,
            # attempt with fallback json_object mode for one retry
            if eff_response_format is not None and eff_response_format.get("type") == "json_schema":
                logger.warning(
                    "json_schema constrained decoding failed for %s; retrying with json_object",
                    skill_filename,
                )
                fallback_response = chat_completion(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                skill_prompt
                                + "\n\nReturn ONLY a valid JSON response matching the required output format."
                            ),
                        },
                        {"role": "user", "content": task_input},
                    ],
                    model=model,
                    api_base=api_base,
                    api_key=api_key,
                    temperature=temperature,
                    timeout=timeout,
                    response_format={"type": "json_object"},
                    n=1,
                    transport=transport,
                    budget=budget,
                    skill=skill_filename,
                    node_id=node_id,
                )
                assert isinstance(fallback_response, dict)
                raw = extract_text_content(fallback_response)
                parsed = extract_json_block(raw)
            else:
                raise
        return raw, parsed

    # n > 1 path: response is a list of single-choice dicts
    assert isinstance(response, list)
    results: List[tuple[str, Any]] = []
    for choice_resp in response:
        raw = extract_text_content(choice_resp)
        try:
            parsed = extract_json_block(raw)
        except LLMError:
            parsed = None
        results.append((raw, parsed))
    return results
