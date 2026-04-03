"""Structured templates for reliable experiment code generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ExperimentTemplate:
    name: str
    description: str
    template: str


TEMPLATES: Dict[str, ExperimentTemplate] = {
    "monte_carlo": ExperimentTemplate(
        name="monte_carlo",
        description="Randomized trials for probabilistic/continuous claims.",
        template=(
            "import json\n"
            "import math\n"
            "import random\n"
            "{USER_IMPORTS}\n\n"
            "{HELPERS}\n\n"
            "def test_instance({PARAMS}):\n"
            "    {TEST_BODY}\n\n"
            "passed_count = 0\n"
            "total = {TRIALS}\n"
            "counterexample = None\n"
            "max_error = 0.0\n"
            "for _trial in range(total):\n"
            "    {INSTANCE_GEN}\n"
            "    _passed, _error, _counterexample = test_instance({ARGS})\n"
            "    if _passed:\n"
            "        passed_count += 1\n"
            "    else:\n"
            "        counterexample = _counterexample\n"
            "        if _error is not None:\n"
            "            max_error = max(max_error, float(_error))\n"
            "        break\n"
            "    if _error is not None:\n"
            "        max_error = max(max_error, float(_error))\n\n"
            "print(json.dumps({\n"
            "    'passed': counterexample is None,\n"
            "    'trials': total,\n"
            "    'max_error': max_error,\n"
            "    'counterexample': counterexample,\n"
            "    'summary': f'{passed_count}/{total} trials passed',\n"
            "}))\n"
        ),
    ),
    "exhaustive_small": ExperimentTemplate(
        name="exhaustive_small",
        description="Exact finite enumeration for small bounded domains.",
        template=(
            "import itertools\n"
            "import json\n"
            "{USER_IMPORTS}\n\n"
            "{HELPERS}\n\n"
            "total = 0\n"
            "counterexample = None\n"
            "max_error = 0.0\n"
            "for {LOOP_VARS} in {ENUMERATOR}:\n"
            "    total += 1\n"
            "    _passed, _error, _counterexample = ({CHECK_EXPR})\n"
            "    if not _passed:\n"
            "        counterexample = _counterexample\n"
            "        if _error is not None:\n"
            "            max_error = max(max_error, float(_error))\n"
            "        break\n"
            "    if _error is not None:\n"
            "        max_error = max(max_error, float(_error))\n\n"
            "print(json.dumps({\n"
            "    'passed': counterexample is None,\n"
            "    'trials': total,\n"
            "    'max_error': max_error,\n"
            "    'counterexample': counterexample,\n"
            "    'summary': f'exhaustive check over {total} assignments',\n"
            "}))\n"
        ),
    ),
    "symbolic_verify": ExperimentTemplate(
        name="symbolic_verify",
        description="Symbolic algebra verification with sympy.",
        template=(
            "import json\n"
            "import sympy as sp\n"
            "{USER_IMPORTS}\n\n"
            "{HELPERS}\n\n"
            "{SYMBOL_SETUP}\n"
            "{EXPR_SETUP}\n"
            "_difference = sp.simplify({LEFT_EXPR} - ({RIGHT_EXPR}))\n"
            "_passed = bool(_difference == 0)\n"
            "print(json.dumps({\n"
            "    'passed': _passed,\n"
            "    'trials': 1,\n"
            "    'max_error': 0.0 if _passed else None,\n"
            "    'counterexample': None if _passed else {'difference': str(_difference)},\n"
            "    'summary': 'symbolic verification completed',\n"
            "}))\n"
        ),
    ),
    "numerical_optimize": ExperimentTemplate(
        name="numerical_optimize",
        description="Numerical search/optimization for extremal constraints.",
        template=(
            "import json\n"
            "import math\n"
            "import numpy as np\n"
            "{USER_IMPORTS}\n\n"
            "{HELPERS}\n\n"
            "{SEARCH_SETUP}\n"
            "_best_value = None\n"
            "_best_witness = None\n"
            "for {LOOP_VARS} in {SEARCH_SPACE}:\n"
            "    _value = ({OBJECTIVE_EXPR})\n"
            "    if _best_value is None or _value {COMPARE_OP} _best_value:\n"
            "        _best_value = _value\n"
            "        _best_witness = {WITNESS_EXPR}\n"
            "_passed = bool({PASS_CONDITION})\n"
            "print(json.dumps({\n"
            "    'passed': _passed,\n"
            "    'trials': int({TRIAL_COUNT_EXPR}),\n"
            "    'max_error': None,\n"
            "    'counterexample': None if _passed else {'witness': _best_witness, 'value': _best_value},\n"
            "    'summary': 'numerical search completed',\n"
            "}))\n"
        ),
    ),
    "measure_compute": ExperimentTemplate(
        name="measure_compute",
        description="Exact rational measure/probability computation.",
        template=(
            "import json\n"
            "from fractions import Fraction\n"
            "{USER_IMPORTS}\n\n"
            "{HELPERS}\n\n"
            "{MEASURE_SETUP}\n"
            "_computed = ({COMPUTED_EXPR})\n"
            "_claimed = ({CLAIMED_EXPR})\n"
            "_passed = bool(_computed == _claimed)\n"
            "print(json.dumps({\n"
            "    'passed': _passed,\n"
            "    'trials': 1,\n"
            "    'max_error': 0.0 if _passed else None,\n"
            "    'counterexample': None if _passed else {'computed_value': str(_computed), 'claimed_value': str(_claimed)},\n"
            "    'summary': 'exact measure computation completed',\n"
            "    'computed_value': str(_computed),\n"
            "    'claimed_value': str(_claimed),\n"
            "}))\n"
        ),
    ),
}


def get_template_catalog() -> str:
    lines = []
    for item in TEMPLATES.values():
        lines.append(f"- {item.name}: {item.description}")
    return "\n".join(lines)


def render_template(template_name: str, slots: dict[str, str]) -> str:
    template = TEMPLATES.get(template_name)
    if template is None:
        raise ValueError(f"Unknown experiment template: {template_name}")
    rendered = template.template
    for key, value in slots.items():
        rendered = rendered.replace("{" + key + "}", value)
    unresolved = [piece for piece in rendered.split("{") if "}" in piece]
    if any(part.split("}", 1)[0].isupper() for part in unresolved):
        # keep hard failure to avoid running partial templates in production.
        raise ValueError("Template rendering left unresolved placeholders.")
    return rendered
