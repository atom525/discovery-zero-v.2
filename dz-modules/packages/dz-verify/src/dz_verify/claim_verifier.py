"""Claim-level verification across quantitative, structural, and heuristic claims."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Union

from gaia.bp.factor_graph import CROMWELL_EPS

from dz_hypergraph.models import HyperGraph
from dz_verify.incremental_codegen import IncrementalCodeGenerator
from dz_hypergraph.tools.experiment_backend import get_experiment_backend, validate_python_code
from dz_hypergraph.tools.lean import verify_proof
from dz_hypergraph.tools.llm import chat_completion, extract_json_block, extract_text_content, load_skill_prompt


@dataclass
class VerifiableClaim:
    claim_text: str
    source_prop_id: Optional[str]
    quantitative: bool
    claim_type: Literal["quantitative", "structural", "heuristic"] = "quantitative"


@dataclass
class ClaimVerificationResult:
    claim: VerifiableClaim
    verdict: Literal["verified", "refuted", "inconclusive"]
    evidence: str
    code: str
    trials: int
    raw_result: dict[str, Any]


# Max claims to verify per plausible action to keep latency bounded.
_MAX_CLAIMS_PER_CALL = 3


class ClaimVerifier:
    def __init__(
        self,
        *,
        backend_name: str = "local",
        default_timeout: int = 60,
        lean_verify_timeout: Optional[int] = None,
        verified_library_path: Optional[Path] = None,
        verification_model: Optional[str] = None,
        max_claims_per_call: int = _MAX_CLAIMS_PER_CALL,
    ) -> None:
        self._backend_name = backend_name
        self._default_timeout = default_timeout
        # Structural claims run verify_proof (lake build); keep separate from
        # default_timeout used for Python experiment execution (often 60s).
        self._lean_verify_timeout = lean_verify_timeout
        self._verification_model = verification_model  # override model for code generation
        self._max_claims_per_call = max_claims_per_call
        self._incremental_codegen = IncrementalCodeGenerator(
            library_path=verified_library_path or Path("./verified_code_library.json"),
            backend_name=backend_name,
        )

    @staticmethod
    def _classify_claim_text(text: str) -> Literal["quantitative", "structural", "heuristic"]:
        cleaned = text.strip()
        if re.search(r"\d|=|<|>|≤|≥|mu\(|μ\(", cleaned):
            return "quantitative"
        lowered = cleaned.casefold()
        if any(
            keyword in lowered
            for keyword in (
                "if ",
                " then ",
                "implies",
                "forall",
                "for all",
                "exists",
                "there exists",
                "lemma",
                "theorem",
                "proof",
                "subgoal",
            )
        ):
            return "structural"
        return "heuristic"

    def extract_claims(self, plausible_output: dict[str, Any]) -> list[VerifiableClaim]:
        claims: list[VerifiableClaim] = []
        seen: set[str] = set()

        def _record(text: str, source_id: Optional[str]) -> None:
            cleaned = text.strip()
            if not cleaned or cleaned in seen:
                return
            claim_type = self._classify_claim_text(cleaned)
            quantitative = claim_type == "quantitative"
            seen.add(cleaned)
            claims.append(
                VerifiableClaim(
                    claim_text=cleaned,
                    source_prop_id=source_id,
                    quantitative=quantitative,
                    claim_type=claim_type,
                )
            )

        for premise in plausible_output.get("premises", []):
            if isinstance(premise, dict):
                _record(str(premise.get("statement", "")), premise.get("id"))

        conclusion = plausible_output.get("conclusion")
        if isinstance(conclusion, dict):
            _record(str(conclusion.get("statement", "")), conclusion.get("id"))
        elif isinstance(conclusion, str):
            _record(conclusion, None)

        for step in plausible_output.get("steps", []):
            if isinstance(step, str):
                _record(step, None)
        return claims

    def generate_verification_code(
        self,
        *,
        claim: VerifiableClaim,
        context: str,
        model: Optional[str] = None,
        record_path: Optional[Path] = None,
    ) -> str:
        prompt = (
            f"Claim:\n{claim.claim_text}\n\n"
            f"Context:\n{context}\n\n"
            "Generate executable Python code that directly verifies or refutes this claim.\n"
            "Return only Python code and print a final JSON line with keys:\n"
            "passed, trials, max_error, counterexample, summary, computed_value, claimed_value."
        )
        try:
            skill_prompt = load_skill_prompt("verify_claim_experiment.skill.md")
        except FileNotFoundError:
            skill_prompt = "Return only executable Python code."
        # Use verification_model if provided (faster code-capable model, e.g. gpt-4o).
        effective_model = self._verification_model or model
        response = chat_completion(
            messages=[
                {"role": "system", "content": skill_prompt},
                {"role": "user", "content": prompt},
            ],
            model=effective_model,
            temperature=0.0,
            stream_record_path=record_path,
        )
        raw = extract_text_content(response)
        code = raw.strip()
        if code.startswith("```"):
            parts = code.splitlines()
            if parts and parts[0].startswith("```"):
                parts = parts[1:]
            if parts and parts[-1].strip() == "```":
                parts = parts[:-1]
            code = "\n".join(parts).strip()
        if code:
            return code
        try:
            parsed_json = extract_json_block(raw)
            if isinstance(parsed_json, dict):
                steps = parsed_json.get("steps", [])
                if isinstance(steps, list) and steps and isinstance(steps[0], str):
                    return steps[0].strip()
        except Exception:
            pass
        fallback = self._incremental_codegen.generate_experiment(
            claim=claim.claim_text,
            context=context,
            model=model,
        )
        if fallback.code.strip():
            return fallback.code
        return raw.strip()

    def verify_claims(
        self,
        *,
        claims: list[VerifiableClaim],
        context: str,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        record_dir: Optional[Path] = None,
    ) -> list[ClaimVerificationResult]:
        # Use dedicated verification_model if set (typically a faster code-capable model),
        # otherwise fall back to the caller-provided model.
        effective_model = self._verification_model or model
        backend = get_experiment_backend(self._backend_name)
        timeout_value = timeout or self._default_timeout
        # Limit to most important claims to keep latency bounded.
        # Prioritise: null-id premises first (new hypotheses), then conclusion, then steps.
        null_id_claims = [c for c in claims if c.source_prop_id is None]
        bound_id_claims = [c for c in claims if c.source_prop_id is not None]
        prioritised = (null_id_claims + bound_id_claims)[: self._max_claims_per_call]
        results: list[ClaimVerificationResult] = []
        for i, claim in enumerate(prioritised, start=1):
            if claim.claim_type == "structural":
                results.append(
                    self._verify_structural_claim(
                        claim=claim,
                        context=context,
                        model=effective_model,
                        timeout_value=timeout_value,
                        record_dir=record_dir,
                        claim_index=i,
                    )
                )
                continue
            if claim.claim_type == "heuristic":
                results.append(
                    self._verify_heuristic_claim(
                        claim=claim,
                        context=context,
                        model=effective_model,
                        record_dir=record_dir,
                        claim_index=i,
                    )
                )
                continue
            prose_path: Optional[Path] = None
            if record_dir is not None:
                record_dir.mkdir(parents=True, exist_ok=True)
                prose_path = record_dir / f"claim_verify_quant_{i}_prose.txt"
            code = self.generate_verification_code(
                claim=claim, context=context, model=effective_model, record_path=prose_path
            )
            if record_dir is not None:
                code_path = record_dir / f"claim_verify_quant_{i}_code.py"
                code_path.write_text(code, encoding="utf-8")
            try:
                validate_python_code(code)
                exec_result = backend.execute(code, timeout=timeout_value)
                stdout = exec_result.stdout or ""
                lines = [line.strip() for line in stdout.splitlines() if line.strip()]
                payload: dict[str, Any]
                if lines:
                    try:
                        payload_any = json.loads(lines[-1])
                        payload = payload_any if isinstance(payload_any, dict) else {}
                    except Exception:
                        payload = {"passed": False, "summary": "execution output was not valid JSON"}
                else:
                    payload = {"passed": False, "summary": "empty execution output"}
                if record_dir is not None:
                    result_path = record_dir / f"claim_verify_quant_{i}_result.txt"
                    result_path.write_text(
                        json.dumps({"stdout": stdout, "stderr": exec_result.stderr, "payload": payload}, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                passed = bool(payload.get("passed", False))
                verdict: Literal["verified", "refuted", "inconclusive"]
                if passed:
                    verdict = "verified"
                elif payload.get("counterexample") is not None:
                    verdict = "refuted"
                else:
                    verdict = "inconclusive"
                results.append(
                    ClaimVerificationResult(
                        claim=claim,
                        verdict=verdict,
                        evidence=str(payload.get("summary", exec_result.stderr[:300])),
                        code=code,
                        trials=int(payload.get("trials", 0) or 0),
                        raw_result=payload,
                    )
                )
            except Exception as exc:
                results.append(
                    ClaimVerificationResult(
                        claim=claim,
                        verdict="inconclusive",
                        evidence=f"verification execution failed: {exc}",
                        code=code,
                        trials=0,
                        raw_result={"passed": False, "error": str(exc)},
                    )
                )
        return results

    def _verify_structural_claim(
        self,
        *,
        claim: VerifiableClaim,
        context: str,
        model: Optional[str],
        timeout_value: int,
        record_dir: Optional[Path] = None,
        claim_index: int = 1,
    ) -> ClaimVerificationResult:
        prompt = (
            "Given the structural mathematical claim and context, generate Lean 4 code that "
            "attempts to type-check and prove the claim without using sorry/admit.\n\n"
            f"Claim:\n{claim.claim_text}\n\n"
            f"Context:\n{context}\n\n"
            "Return Lean code only."
        )
        try:
            skill_prompt = load_skill_prompt("lean_proof.skill.md")
        except FileNotFoundError:
            skill_prompt = "Return Lean code only."
        prose_path: Optional[Path] = None
        if record_dir is not None:
            record_dir.mkdir(parents=True, exist_ok=True)
            prose_path = record_dir / f"claim_verify_struct_{claim_index}_prose.txt"
        response = chat_completion(
            messages=[
                {"role": "system", "content": skill_prompt},
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=0.0,
            stream_record_path=prose_path,
        )
        raw = extract_text_content(response).strip()
        lean_code = raw
        if lean_code.startswith("```"):
            parts = lean_code.splitlines()
            if parts and parts[0].startswith("```"):
                parts = parts[1:]
            if parts and parts[-1].strip() == "```":
                parts = parts[:-1]
            lean_code = "\n".join(parts).strip()
        if not lean_code:
            return ClaimVerificationResult(
                claim=claim,
                verdict="inconclusive",
                evidence="empty lean proof candidate",
                code="",
                trials=0,
                raw_result={"error": "empty lean proof candidate"},
            )
        if record_dir is not None:
            lean_file = record_dir / f"claim_verify_struct_{claim_index}_code.lean"
            lean_file.write_text(lean_code, encoding="utf-8")
        lean_t = (
            self._lean_verify_timeout
            if self._lean_verify_timeout is not None
            else timeout_value
        )
        verify = verify_proof(lean_code, timeout=lean_t)
        verdict: Literal["verified", "refuted", "inconclusive"]
        if verify.success:
            verdict = "verified"
            evidence = "Lean verification succeeded."
        else:
            verdict = "inconclusive"
            evidence = verify.error_message or "Lean verification failed."
        return ClaimVerificationResult(
            claim=claim,
            verdict=verdict,
            evidence=evidence,
            code=lean_code,
            trials=1,
            raw_result={
                "success": verify.success,
                "error_message": verify.error_message,
                "stderr": verify.stderr,
                "stdout": verify.stdout,
                "exit_code": verify.exit_code,
            },
        )

    def _verify_heuristic_claim(
        self,
        *,
        claim: VerifiableClaim,
        context: str,
        model: Optional[str],
        record_dir: Optional[Path] = None,
        claim_index: int = 1,
    ) -> ClaimVerificationResult:
        prompt = (
            "Assess whether the heuristic claim is plausible based on the provided context.\n"
            "Return JSON with fields: verdict (verified|refuted|inconclusive), evidence.\n\n"
            f"Claim:\n{claim.claim_text}\n\n"
            f"Context:\n{context}\n"
        )
        prose_path: Optional[Path] = None
        if record_dir is not None:
            record_dir.mkdir(parents=True, exist_ok=True)
            prose_path = record_dir / f"claim_verify_heuristic_{claim_index}_prose.txt"
        response = chat_completion(
            messages=[
                {"role": "system", "content": "You are a strict mathematical claim judge. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=0.0,
            stream_record_path=prose_path,
        )
        raw = extract_text_content(response)
        verdict: Literal["verified", "refuted", "inconclusive"] = "inconclusive"
        evidence = "Heuristic judgement unavailable."
        try:
            parsed = extract_json_block(raw)
            parsed_verdict = str(parsed.get("verdict", "")).strip().lower()
            if parsed_verdict in {"verified", "refuted", "inconclusive"}:
                verdict = parsed_verdict  # type: ignore[assignment]
            evidence = str(parsed.get("evidence", evidence))
        except Exception:
            evidence = raw[:300] if raw else evidence
        return ClaimVerificationResult(
            claim=claim,
            verdict=verdict,
            evidence=evidence,
            code="",
            trials=1,
            raw_result={"raw": raw},
        )

    def update_graph_beliefs(
        self,
        *,
        graph: HyperGraph,
        results: list[ClaimVerificationResult],
        positive_delta: float = 0.3,
    ) -> int:
        updated = 0
        for item in results:
            claim_text = item.claim.claim_text
            matches = graph.find_node_ids_by_statement(claim_text)
            if len(matches) != 1:
                continue
            node = graph.nodes[matches[0]]
            if node.is_locked():
                continue
            if item.verdict == "verified":
                node.prior = min(1.0 - CROMWELL_EPS, node.prior + positive_delta)
                updated += 1
            elif item.verdict == "refuted":
                node.prior = max(CROMWELL_EPS, node.prior * 0.1)
                updated += 1
        return updated
