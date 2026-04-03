"""
Token budget management for LLM calls.

Tracks cumulative token usage across a discovery run and enforces
configurable hard limits, enabling graceful degradation when budgets
are exhausted rather than running indefinitely.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class BudgetExhaustedError(Exception):
    """Raised when a token budget limit is exceeded."""

    def __init__(self, message: str, *, budget_type: str = "", used: int = 0, limit: int = 0):
        super().__init__(message)
        self.budget_type = budget_type
        self.used = used
        self.limit = limit


@dataclass
class TokenUsageRecord:
    """Record of token usage for a single LLM call."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    skill: str = ""
    node_id: str = ""


@dataclass
class TokenBudget:
    """
    Thread-safe token budget tracker with optional enforcement.

    Per-run budget tracks all calls within a discovery run.
    Per-action budget (per_action_prompt / per_action_completion) is checked
    before each individual call and resets are caller-managed.

    Set limit_* to 0 (or None) to disable enforcement for that dimension.
    """

    # Cumulative usage
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    call_count: int = 0

    # Hard limits (0 = unlimited)
    limit_prompt: int = 0
    limit_completion: int = 0
    limit_calls: int = 0

    # Per-action (per single skill call) limits
    per_action_prompt: int = 0
    per_action_completion: int = 0

    # Usage history for diagnostics
    history: list[TokenUsageRecord] = field(default_factory=list, repr=False)

    # Internal
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    def record(
        self,
        usage: Dict[str, Any],
        *,
        model: str = "",
        skill: str = "",
        node_id: str = "",
    ) -> None:
        """Record token usage from an API response's usage dict."""
        with self._lock:
            prompt = int(usage.get("prompt_tokens", 0))
            completion = int(usage.get("completion_tokens", 0))
            total = int(usage.get("total_tokens", prompt + completion))

            self.total_prompt_tokens += prompt
            self.total_completion_tokens += completion
            self.call_count += 1

            self.history.append(
                TokenUsageRecord(
                    model=model,
                    prompt_tokens=prompt,
                    completion_tokens=completion,
                    total_tokens=total,
                    skill=skill,
                    node_id=node_id,
                )
            )

    def check_before_call(
        self,
        *,
        estimated_prompt: int = 0,
        estimated_completion: int = 0,
        skill: str = "",
        node_id: str = "",
    ) -> None:
        """
        Check budgets *before* making a call.

        Raises BudgetExhaustedError if the estimated usage would exceed a
        hard limit.  Pass estimated_* = 0 to skip estimation checks.
        """
        with self._lock:
            if self.limit_calls and self.call_count >= self.limit_calls:
                raise BudgetExhaustedError(
                    f"Token budget exhausted: call count {self.call_count} >= limit {self.limit_calls}",
                    budget_type="calls",
                    used=self.call_count,
                    limit=self.limit_calls,
                )

            if self.limit_prompt and estimated_prompt:
                projected = self.total_prompt_tokens + estimated_prompt
                if projected > self.limit_prompt:
                    raise BudgetExhaustedError(
                        f"Prompt token budget would be exceeded: {projected} > {self.limit_prompt}",
                        budget_type="prompt",
                        used=self.total_prompt_tokens,
                        limit=self.limit_prompt,
                    )

            if self.limit_completion and estimated_completion:
                projected = self.total_completion_tokens + estimated_completion
                if projected > self.limit_completion:
                    raise BudgetExhaustedError(
                        f"Completion token budget would be exceeded: {projected} > {self.limit_completion}",
                        budget_type="completion",
                        used=self.total_completion_tokens,
                        limit=self.limit_completion,
                    )

    def exhausted(self) -> bool:
        """Return True if any hard limit has been reached."""
        with self._lock:
            if self.limit_calls and self.call_count >= self.limit_calls:
                return True
            if self.limit_prompt and self.total_prompt_tokens >= self.limit_prompt:
                return True
            if self.limit_completion and self.total_completion_tokens >= self.limit_completion:
                return True
            return False

    def remaining(self) -> Dict[str, Optional[int]]:
        """Return remaining budget for each dimension (None = unlimited)."""
        with self._lock:
            return {
                "prompt": (
                    max(0, self.limit_prompt - self.total_prompt_tokens)
                    if self.limit_prompt else None
                ),
                "completion": (
                    max(0, self.limit_completion - self.total_completion_tokens)
                    if self.limit_completion else None
                ),
                "calls": (
                    max(0, self.limit_calls - self.call_count)
                    if self.limit_calls else None
                ),
            }

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary for logging / benchmark reports."""
        with self._lock:
            return {
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_tokens,
                "call_count": self.call_count,
                "limits": {
                    "prompt": self.limit_prompt or None,
                    "completion": self.limit_completion or None,
                    "calls": self.limit_calls or None,
                },
                "remaining": self.remaining(),
            }

    def reset_action_counters(self) -> None:
        """Reset per-action counters (if tracking per-action totals externally)."""
        # Future: could track per-action accumulators here
        pass

    def __repr__(self) -> str:
        return (
            f"TokenBudget(prompt={self.total_prompt_tokens}/{self.limit_prompt or '∞'}, "
            f"completion={self.total_completion_tokens}/{self.limit_completion or '∞'}, "
            f"calls={self.call_count}/{self.limit_calls or '∞'})"
        )
