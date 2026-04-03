"""
Evolutionary synthesis for experiment programs.

The first production version stays deliberately simple:
  - generate a diverse initial population using multiple prompt strategies
  - execute candidates through the existing experiment backend
  - keep the highest-fitness programs and mutate them with lightweight LLM edits
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from dz_hypergraph.tools.experiment_backend import (
    CodeValidationError,
    ExperimentResult,
    get_experiment_backend,
)
from dz_hypergraph.tools.llm import run_skill


PROMPT_STRATEGIES: tuple[str, ...] = (
    "exhaustive_enumeration",
    "boundary_testing",
    "random_search",
    "algebraic_verification",
    "statistical_analysis",
)


@dataclass
class ExperimentProgram:
    code: str
    strategy: str
    generation: int = 0
    fitness: float = 0.0
    result: Optional[dict] = None


@dataclass
class EvolutionConfig:
    population_size: int = 5
    generations: int = 2
    retain_top_k: int = 2
    timeout_seconds: int = 90
    backend: str = "docker"


class ExperimentEvolver:
    def __init__(self, config: Optional[EvolutionConfig] = None) -> None:
        self.config = config or EvolutionConfig()

    def evolve(
        self,
        *,
        conjecture: str,
        context: str = "",
        model: Optional[str] = None,
    ) -> list[ExperimentProgram]:
        population = self._initial_population(conjecture=conjecture, context=context, model=model)
        if not population:
            return []
        for generation in range(self.config.generations):
            for program in population:
                if program.result is None:
                    self._evaluate(program)
            population.sort(key=lambda item: item.fitness, reverse=True)
            elites = population[: self.config.retain_top_k]
            if generation + 1 >= self.config.generations:
                population = elites
                break
            mutants: list[ExperimentProgram] = list(elites)
            for elite in elites:
                mutated = self._mutate(
                    elite,
                    conjecture=conjecture,
                    context=context,
                    model=model,
                    generation=generation + 1,
                )
                if mutated is not None:
                    mutants.append(mutated)
            population = mutants[: self.config.population_size]
        population.sort(key=lambda item: item.fitness, reverse=True)
        return population

    def _initial_population(
        self,
        *,
        conjecture: str,
        context: str,
        model: Optional[str],
    ) -> list[ExperimentProgram]:
        population: list[ExperimentProgram] = []
        for strategy in PROMPT_STRATEGIES[: self.config.population_size]:
            code = self._generate_program(
                conjecture=conjecture,
                context=context,
                strategy=strategy,
                model=model,
            )
            if code:
                population.append(ExperimentProgram(code=code, strategy=strategy))
        return population

    def _generate_program(
        self,
        *,
        conjecture: str,
        context: str,
        strategy: str,
        model: Optional[str],
    ) -> str:
        task_input = "\n".join(
            [
                "Write a scientific Python experiment for the conjecture below.",
                "Return JSON with key `code` only.",
                "The code must print exactly one final JSON line with keys:",
                "`passed`, `trials`, `max_error`, `counterexample`, `summary`.",
                f"Strategy: {strategy}",
                f"Conjecture: {conjecture}",
                f"Context: {context or '(none)'}",
            ]
        )
        try:
            _raw, parsed = run_skill("experiment_evolution_mutate.skill.md", task_input, model=model)
        except Exception:
            return ""
        if not isinstance(parsed, dict):
            return ""
        return str(parsed.get("code", "")).strip()

    def _mutate(
        self,
        parent: ExperimentProgram,
        *,
        conjecture: str,
        context: str,
        model: Optional[str],
        generation: int,
    ) -> Optional[ExperimentProgram]:
        task_input = "\n".join(
            [
                "Mutate the following experiment code to improve coverage, exactness, or robustness.",
                "Return JSON with key `code` only.",
                f"Conjecture: {conjecture}",
                f"Context: {context or '(none)'}",
                f"Parent strategy: {parent.strategy}",
                "Parent code:",
                parent.code,
            ]
        )
        try:
            _raw, parsed = run_skill("experiment_evolution_mutate.skill.md", task_input, model=model)
        except Exception:
            return None
        if not isinstance(parsed, dict):
            return None
        code = str(parsed.get("code", "")).strip()
        if not code:
            return None
        return ExperimentProgram(
            code=code,
            strategy=f"mutated:{parent.strategy}",
            generation=generation,
        )

    def _evaluate(self, program: ExperimentProgram) -> None:
        backend = get_experiment_backend(self.config.backend)
        try:
            result = backend.execute(program.code, timeout=self.config.timeout_seconds)
        except CodeValidationError as exc:
            program.result = {"success": False, "error": str(exc)}
            program.fitness = -1.0
            return
        program.result = result.to_dict()
        program.fitness = self._fitness(result)

    def _fitness(self, result: ExperimentResult) -> float:
        if not result.success:
            return -0.5
        parsed = result.parsed_json if isinstance(result.parsed_json, dict) else {}
        passed = 1.0 if parsed.get("passed") else 0.0
        trials = min(1.0, float(parsed.get("trials", 0)) / 100.0)
        counterexample_bonus = 0.4 if parsed.get("counterexample") else 0.0
        exact_bonus = 0.2 if "fraction" in result.stdout.lower() or "sympy" in result.stdout.lower() else 0.0
        speed_bonus = max(0.0, 1.0 - result.execution_time_ms / 10000.0)
        return 0.4 * passed + 0.2 * trials + counterexample_bonus + exact_bonus + 0.2 * speed_bonus
