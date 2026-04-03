"""Incremental experiment code generation with repair and verification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dz_hypergraph.tools.experiment_backend import get_experiment_backend, validate_python_code
from dz_hypergraph.tools.experiment_templates import get_template_catalog, render_template
from dz_hypergraph.tools.llm import extract_json_block, extract_text_content, chat_completion, run_skill
from dz_hypergraph.tools.verified_code_library import VerifiedCodeLibrary


@dataclass
class IncrementalCodegenResult:
    code: str
    success: bool
    summary: str


class IncrementalCodeGenerator:
    def __init__(self, *, library_path: Path, backend_name: str = "local") -> None:
        self.code_library = VerifiedCodeLibrary(library_path)
        self.backend_name = backend_name

    def _generate_skeleton(self, claim: str, context: str, model: Optional[str]) -> dict:
        prompt = (
            f"Claim:\n{claim}\n\n"
            f"Context:\n{context}\n\n"
            "Return JSON with keys: functions (list of {name, signature, purpose}) and template_hint."
        )
        raw, parsed = run_skill("claim_verification.skill.md", prompt, model=model)
        if isinstance(parsed, dict):
            return parsed
        return {"functions": [{"name": "test_instance", "signature": "()", "purpose": raw[:200]}], "template_hint": "monte_carlo"}

    def _fill_template(self, claim: str, context: str, skeleton: dict, model: Optional[str]) -> str:
        template_prompt = (
            f"Claim:\n{claim}\n\nContext:\n{context}\n\n"
            f"Template catalog:\n{get_template_catalog()}\n\n"
            f"Skeleton:\n{json.dumps(skeleton, ensure_ascii=False)}"
        )
        _raw, parsed = run_skill("fill_experiment_template.skill.md", template_prompt, model=model)
        if not isinstance(parsed, dict):
            raise ValueError("Template fill skill did not return a JSON object.")
        template = str(parsed.get("template", "")).strip()
        slots_raw = parsed.get("slots", {})
        if not isinstance(slots_raw, dict):
            raise ValueError("Template fill skill did not return slots dictionary.")
        slots = {str(k): str(v) for k, v in slots_raw.items()}
        return render_template(template, slots)

    def _repair_code(self, code: str, error: str, model: Optional[str]) -> str:
        prompt = (
            "Fix the Python code and return ONLY corrected Python code.\n"
            f"Error:\n{error}\n\nCode:\n{code}"
        )
        resp = chat_completion(
            messages=[
                {"role": "system", "content": "Return only Python code."},
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=0.0,
        )
        fixed = extract_text_content(resp).strip()
        if fixed.startswith("```"):
            fixed = fixed.split("```", 1)[-1]
            fixed = fixed.rsplit("```", 1)[0]
        return fixed.strip()

    def _dry_run(self, code: str, timeout: int = 15) -> tuple[bool, str]:
        validate_python_code(code)
        backend = get_experiment_backend(self.backend_name)
        result = backend.execute(code, timeout=timeout)
        if result.timed_out:
            return False, "timed out"
        if not result.success:
            return False, result.error_message or result.stderr[:300]
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            return False, "empty stdout"
        try:
            payload = json.loads(lines[-1])
            if not isinstance(payload, dict) or "passed" not in payload:
                return False, "invalid JSON summary"
        except Exception as exc:
            return False, f"invalid JSON summary: {exc}"
        return True, "ok"

    def generate_experiment(self, claim: str, context: str, model: Optional[str]) -> IncrementalCodegenResult:
        skeleton = self._generate_skeleton(claim, context, model)
        code = self._fill_template(claim, context, skeleton, model)
        helpers = self.code_library.retrieve_relevant(claim, top_k=3)
        if helpers:
            helper_block = "\n\n".join(item.code for item in helpers)
            code = helper_block + "\n\n" + code
        success, reason = self._dry_run(code)
        retries = 0
        while not success and retries < 3:
            code = self._repair_code(code, reason, model)
            success, reason = self._dry_run(code)
            retries += 1
        if success:
            self.code_library.add_verified(
                func_name=f"auto_helper_{abs(hash(claim)) % 100000}",
                code=code,
                signature="generated_experiment()",
                docstring=f"Generated helper for claim: {claim[:120]}",
                success=True,
                summary="incremental_codegen verified",
            )
        return IncrementalCodegenResult(code=code, success=success, summary=reason)
