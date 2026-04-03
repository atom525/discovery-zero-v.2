"""
Failure Router — adaptive recovery and replanning for Discovery Zero.

When an action (plausible/experiment/lean/bridge_plan/decompose) fails,
the FailureRouter classifies the error and determines the best recovery
strategy: retry, switch module, replan with feedback, skip, or escalate.

The rule matrix is informed by the diagnostic patterns observed in the
evaluation runs:
  - json_parse / api_502: usually transient → retry (up to 3x) then switch
  - validation (bridge plan schema): deterministic → replan with feedback
  - lean_error (syntax): Lean compiler issue → log and replan
  - lean_error (logic): proof attempt failed → retry with Lean feedback
  - timeout: resource issue → retry once with extended timeout, then skip
  - experiment_refuted: logical refutation → mark refuted + propagate
  - consecutive_module_failures: 3+ failures → escalate (double UCB penalty)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from dz_hypergraph.models import Module

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Error type taxonomy                                                  #
# ------------------------------------------------------------------ #

class ErrorType:
    """Canonical error type strings used in FailureRecord."""
    JSON_PARSE = "json_parse"
    API_502 = "api_502"
    API_TIMEOUT = "api_timeout"
    API_ERROR = "api_error"
    VALIDATION = "validation"
    LEAN_SYNTAX = "lean_syntax"
    LEAN_LOGIC = "lean_logic"
    LEAN_TIMEOUT = "lean_timeout"
    LEAN_COMPILE = "lean_compile"
    EXPERIMENT_REFUTED = "experiment_refuted"
    EXPERIMENT_TIMEOUT = "experiment_timeout"
    EXPERIMENT_ERROR = "experiment_error"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NODE_NOT_FOUND = "node_not_found"
    UNKNOWN = "unknown"


class RecoveryAction:
    """Canonical recovery action strings."""
    RETRY = "retry"
    RETRY_EXTENDED_TIMEOUT = "retry_extended_timeout"
    REPLAN_WITH_FEEDBACK = "replan_with_feedback"
    SWITCH_MODULE = "switch_module"
    SKIP = "skip"
    ESCALATE = "escalate"
    MARK_REFUTED = "mark_refuted"


# ------------------------------------------------------------------ #
# Failure record                                                       #
# ------------------------------------------------------------------ #

@dataclass
class FailureRecord:
    """One failed action attempt."""

    node_id: str
    module: Module
    stage: str
    """Stage within the module: e.g. 'plausible', 'bridge_plan', 'lean_verify'."""

    error_type: str
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    recovery_action: Optional[str] = None
    attempt_number: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "module": self.module.value,
            "stage": self.stage,
            "error_type": self.error_type,
            "message": self.message[:300],
            "timestamp": self.timestamp.isoformat(),
            "recovery_action": self.recovery_action,
            "attempt_number": self.attempt_number,
        }


# ------------------------------------------------------------------ #
# Failure router                                                       #
# ------------------------------------------------------------------ #

class FailureRouter:
    """
    Classifies failures and determines recovery actions.

    The routing logic is a rule matrix that maps (error_type, consecutive_failures)
    → recovery_action.  The matrix is designed to:
      1. Retry transient errors (API 502, JSON parse) quickly.
      2. Replan with feedback for validation/schema errors (deterministic).
      3. Switch modules when one module consistently fails.
      4. Skip when no progress is possible (budget, unsolvable).
      5. Escalate (double UCB penalty + notify engine) for persistent failures.
    """

    def __init__(
        self,
        max_retries_per_error_type: int = 3,
        max_consecutive_failures_before_escalate: int = 3,
    ) -> None:
        self._max_retries = max_retries_per_error_type
        self._escalate_threshold = max_consecutive_failures_before_escalate

        # Per-(node_id, module, error_type) retry counts
        self._retry_counts: Dict[tuple, int] = {}

        # Per-(node_id, module) consecutive failure counts
        self._consecutive: Dict[tuple, int] = {}

        self._history: List[FailureRecord] = []

    def route(
        self,
        record: FailureRecord,
    ) -> str:
        """
        Classify a failure and return a recovery action string.

        Side effects:
          - Updates retry counts and consecutive failure memory.
          - Appends to history.
        """
        self._history.append(record)

        node_key = (record.node_id, record.module.value)
        err_key = (record.node_id, record.module.value, record.error_type)

        # Update counters
        consec = self._consecutive.get(node_key, 0) + 1
        self._consecutive[node_key] = consec
        retries = self._retry_counts.get(err_key, 0) + 1
        self._retry_counts[err_key] = retries

        action = self._rule_matrix(record, consec, retries)
        record.recovery_action = action

        logger.info(
            "FailureRouter: node=%s module=%s error_type=%s consec=%d retries=%d → %s",
            record.node_id[:12],
            record.module.value,
            record.error_type,
            consec,
            retries,
            action,
        )
        return action

    def record_success(self, node_id: str, module: Module) -> None:
        """Reset consecutive failure counter after a successful action."""
        self._consecutive[(node_id, module.value)] = 0

    def consecutive_failures(self, node_id: str, module: Module) -> int:
        return self._consecutive.get((node_id, module.value), 0)

    def retry_count(self, node_id: str, module: Module, error_type: str) -> int:
        return self._retry_counts.get((node_id, module.value, error_type), 0)

    def history_for_node(self, node_id: str) -> List[FailureRecord]:
        return [r for r in self._history if r.node_id == node_id]

    # ---------------------------------------------------------------- #
    # Rule matrix                                                       #
    # ---------------------------------------------------------------- #

    def _rule_matrix(
        self,
        record: FailureRecord,
        consecutive_failures: int,
        retries_this_error: int,
    ) -> str:
        et = record.error_type

        # Budget exhausted — skip immediately
        if et == ErrorType.BUDGET_EXHAUSTED:
            return RecoveryAction.SKIP

        # Escalate if consecutive failures exceed threshold
        if consecutive_failures >= self._escalate_threshold:
            return RecoveryAction.ESCALATE

        # ---- Transient API errors ----
        if et in (ErrorType.JSON_PARSE, ErrorType.API_502, ErrorType.API_ERROR):
            if retries_this_error <= self._max_retries:
                return RecoveryAction.RETRY
            return RecoveryAction.SWITCH_MODULE

        # ---- API timeout ----
        if et == ErrorType.API_TIMEOUT:
            if retries_this_error == 1:
                return RecoveryAction.RETRY_EXTENDED_TIMEOUT
            return RecoveryAction.SWITCH_MODULE

        # ---- Validation / schema errors (deterministic) ----
        if et == ErrorType.VALIDATION:
            if retries_this_error <= 2:
                return RecoveryAction.REPLAN_WITH_FEEDBACK
            return RecoveryAction.SWITCH_MODULE

        # ---- Lean errors ----
        if et == ErrorType.LEAN_SYNTAX:
            if retries_this_error <= 2:
                return RecoveryAction.REPLAN_WITH_FEEDBACK
            return RecoveryAction.SWITCH_MODULE

        if et == ErrorType.LEAN_LOGIC:
            if retries_this_error <= self._max_retries:
                return RecoveryAction.REPLAN_WITH_FEEDBACK
            return RecoveryAction.SKIP

        if et in (ErrorType.LEAN_COMPILE, ErrorType.LEAN_TIMEOUT):
            if retries_this_error == 1:
                return RecoveryAction.RETRY_EXTENDED_TIMEOUT
            return RecoveryAction.SKIP

        # ---- Experiment errors ----
        if et == ErrorType.EXPERIMENT_REFUTED:
            # Experiment refutations are handled via the weakened ingest path;
            # no separate "mark refuted" action is needed.  Switch module so
            # the engine tries a different approach on the next iteration.
            return RecoveryAction.SWITCH_MODULE

        if et == ErrorType.EXPERIMENT_TIMEOUT:
            if retries_this_error == 1:
                return RecoveryAction.RETRY_EXTENDED_TIMEOUT
            return RecoveryAction.SWITCH_MODULE

        if et == ErrorType.EXPERIMENT_ERROR:
            if retries_this_error <= 2:
                return RecoveryAction.RETRY
            return RecoveryAction.SWITCH_MODULE

        # ---- Default ----
        if retries_this_error <= 2:
            return RecoveryAction.RETRY
        return RecoveryAction.SKIP


def classify_error(exc: Exception, stage: str) -> str:
    """
    Classify an exception into a canonical ErrorType string.

    This is a best-effort heuristic — callers can override.
    """
    msg = str(exc).lower()

    if "budget" in msg or "exhausted" in msg:
        return ErrorType.BUDGET_EXHAUSTED

    if "502" in msg or "bad gateway" in msg:
        return ErrorType.API_502

    if "timeout" in msg and "lean" in stage.lower():
        return ErrorType.LEAN_TIMEOUT

    if "timeout" in msg and "experiment" in stage.lower():
        return ErrorType.EXPERIMENT_TIMEOUT

    if "timeout" in msg:
        return ErrorType.API_TIMEOUT

    if "json" in msg or "parse" in msg or "decode" in msg:
        return ErrorType.JSON_PARSE

    if "validation" in msg or "schema" in msg or "required field" in msg:
        return ErrorType.VALIDATION

    if "lean" in stage.lower() or "lean" in msg:
        if "syntax" in msg or "parse error" in msg or "unexpected token" in msg:
            return ErrorType.LEAN_SYNTAX
        if "goals" in msg or "failed to prove" in msg or "type mismatch" in msg:
            return ErrorType.LEAN_LOGIC
        return ErrorType.LEAN_COMPILE

    if "experiment" in stage.lower():
        if "refut" in msg:
            return ErrorType.EXPERIMENT_REFUTED
        return ErrorType.EXPERIMENT_ERROR

    if "502" in msg or "503" in msg or "504" in msg or "429" in msg:
        return ErrorType.API_ERROR

    return ErrorType.UNKNOWN
