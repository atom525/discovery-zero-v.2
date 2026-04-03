"""
Unified configuration for Discovery Zero.

All runtime constants are centralised here. Settings are read from environment
variables (prefix: DISCOVERY_ZERO_) and optionally from a .env.local file in
the project root. Every other module imports from this file instead of
hard-coding values.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


_PROJECT_ROOT: Optional[Path] = None


def _get_project_root() -> Path:
    global _PROJECT_ROOT
    if _PROJECT_ROOT is None:
        _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    return _PROJECT_ROOT


def _load_env_file_once() -> None:
    """Load .env.local from project root once at first import."""
    env_path = _get_project_root() / ".env.local"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_env_file_once()


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key, "").strip()
    return int(v) if v else default


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key, "").strip()
    return float(v) if v else default


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


class ZeroConfig:
    """
    Centralised configuration for all Discovery Zero modules.

    All values can be overridden via DISCOVERY_ZERO_* environment variables
    or via .env.local.
    """

    # ------------------------------------------------------------------ #
    # LLM / API                                                           #
    # ------------------------------------------------------------------ #
    llm_model: str = _env("DISCOVERY_ZERO_LLM_MODEL") or _env("LITELLM_PROXY_MODEL", "cds/Claude-4.6-opus")
    llm_api_base: str = _env("LITELLM_PROXY_API_BASE")
    llm_api_key: str = _env("LITELLM_PROXY_API_KEY")

    # Per-request timeout in seconds
    llm_timeout: int = _env_int("DISCOVERY_ZERO_LLM_TIMEOUT", 300)

    # httpx transport settings
    llm_max_retries: int = _env_int("DISCOVERY_ZERO_LLM_MAX_RETRIES", 4)
    llm_retry_base_delay: float = _env_float("DISCOVERY_ZERO_LLM_RETRY_BASE_DELAY", 1.0)
    llm_retry_max_delay: float = _env_float("DISCOVERY_ZERO_LLM_RETRY_MAX_DELAY", 60.0)
    llm_retry_jitter: float = _env_float("DISCOVERY_ZERO_LLM_RETRY_JITTER", 0.5)
    llm_connect_timeout: float = _env_float("DISCOVERY_ZERO_LLM_CONNECT_TIMEOUT", 10.0)
    llm_pool_max_connections: int = _env_int("DISCOVERY_ZERO_LLM_POOL_MAX_CONNECTIONS", 10)
    llm_stream_chunk_timeout: float = _env_float("DISCOVERY_ZERO_LLM_STREAM_CHUNK_TIMEOUT", 120.0)
    llm_stream_wall_timeout: float = _env_float("DISCOVERY_ZERO_LLM_STREAM_WALL_TIMEOUT", 900.0)

    # Max output tokens per LLM request. 0 = no limit (use model default).
    llm_max_output_tokens: int = _env_int("DISCOVERY_ZERO_MAX_OUTPUT_TOKENS", 16000)

    # Auto-continue when model hits output limit (finish_reason="length").
    # Value = max number of continuation requests (0 = disabled).
    llm_auto_continue: int = _env_int("DISCOVERY_ZERO_LLM_AUTO_CONTINUE", 3)

    # Enable response_format: json_object / json_schema constrained decoding
    llm_structured_output: bool = _env_bool("DISCOVERY_ZERO_LLM_STRUCTURED_OUTPUT", True)

    # Prefer streaming chat completions for long-running requests to avoid
    # read timeouts while the model is still generating tokens.
    llm_streaming: bool = _env_bool("DISCOVERY_ZERO_LLM_STREAMING", True)

    # Enable incremental (multi-step) skill generation for complex outputs
    llm_incremental_skills: bool = _env_bool("DISCOVERY_ZERO_LLM_INCREMENTAL_SKILLS", True)

    # Max self-correction attempts when skill output fails validation
    llm_self_correction_attempts: int = _env_int("DISCOVERY_ZERO_LLM_SELF_CORRECTION_ATTEMPTS", 3)

    # ------------------------------------------------------------------ #
    # Belief Propagation                                                  #
    # ------------------------------------------------------------------ #
    bp_backend: str = _env("DISCOVERY_ZERO_BP_BACKEND", "gaia_v2")  # "gaia" | "gaia_v2" | "energy"
    bp_max_iterations: int = _env_int("DISCOVERY_ZERO_BP_MAX_ITERATIONS", 50)
    bp_damping: float = _env_float("DISCOVERY_ZERO_BP_DAMPING", 0.5)
    bp_tolerance: float = _env_float("DISCOVERY_ZERO_BP_TOLERANCE", 1e-6)

    # Incremental BP: only re-propagate subgraph affected by changed edges
    bp_incremental: bool = _env_bool("DISCOVERY_ZERO_BP_INCREMENTAL", True)

    # ------------------------------------------------------------------ #
    # Search Engine                                                       #
    # ------------------------------------------------------------------ #
    search_c_explore: float = _env_float("DISCOVERY_ZERO_SEARCH_C_EXPLORE", 1.4)
    search_max_frontiers: int = _env_int("DISCOVERY_ZERO_SEARCH_MAX_FRONTIERS", 5)
    search_diversity_weight: float = _env_float("DISCOVERY_ZERO_SEARCH_DIVERSITY_WEIGHT", 0.3)
    search_virtual_loss: float = _env_float("DISCOVERY_ZERO_SEARCH_VIRTUAL_LOSS", 1.0)
    search_progressive_widening_base: float = _env_float("DISCOVERY_ZERO_SEARCH_PW_BASE", 1.5)

    # Enable RMaxTS intrinsic rewards
    search_intrinsic_rewards: bool = _env_bool("DISCOVERY_ZERO_SEARCH_INTRINSIC_REWARDS", False)
    search_intrinsic_belief_weight: float = _env_float("DISCOVERY_ZERO_SEARCH_INTR_BELIEF_W", 0.5)
    search_intrinsic_novelty_weight: float = _env_float("DISCOVERY_ZERO_SEARCH_INTR_NOVELTY_W", 0.3)
    search_intrinsic_surprise_weight: float = _env_float("DISCOVERY_ZERO_SEARCH_INTR_SURPRISE_W", 0.2)

    # ------------------------------------------------------------------ #
    # Discovery Engine                                                    #
    # ------------------------------------------------------------------ #
    engine_max_rounds: int = _env_int("DISCOVERY_ZERO_ENGINE_MAX_ROUNDS", 20)
    engine_max_concurrent_actions: int = _env_int("DISCOVERY_ZERO_ENGINE_MAX_CONCURRENT", 3)
    engine_bridge_max_passes: int = _env_int("DISCOVERY_ZERO_ENGINE_BRIDGE_MAX_PASSES", 5)

    # Plausible planning max retry attempts
    engine_plausible_max_attempts: int = _env_int("DISCOVERY_ZERO_ENGINE_PLAUSIBLE_MAX_ATTEMPTS", 4)
    engine_bridge_max_attempts: int = _env_int("DISCOVERY_ZERO_ENGINE_BRIDGE_MAX_ATTEMPTS", 3)
    max_bridge_followup_rounds: int = _env_int("DISCOVERY_ZERO_MAX_BRIDGE_FOLLOWUP_ROUNDS", 5)
    engine_lean_max_attempts: int = _env_int("DISCOVERY_ZERO_ENGINE_LEAN_MAX_ATTEMPTS", 3)

    # Min acceptable plausible confidence before replanning
    engine_min_plausible_confidence: float = _env_float(
        "DISCOVERY_ZERO_ENGINE_MIN_PLAUSIBLE_CONFIDENCE", 0.45
    )

    # Max number of nodes included in graph context for LLM prompts
    engine_context_max_nodes: int = _env_int("DISCOVERY_ZERO_ENGINE_CONTEXT_MAX_NODES", 12)

    # ------------------------------------------------------------------ #
    # Timeouts (seconds)                                                  #
    # ------------------------------------------------------------------ #
    experiment_timeout: int = _env_int("DISCOVERY_ZERO_EXPERIMENT_TIMEOUT", 120)
    experiment_code_timeout: int = _env_int("DISCOVERY_ZERO_EXPERIMENT_CODE_TIMEOUT", 60)
    lean_timeout: int = _env_int("DISCOVERY_ZERO_LEAN_TIMEOUT", 300)
    decompose_timeout: int = _env_int("DISCOVERY_ZERO_DECOMPOSE_TIMEOUT", 180)
    tactic_prover_timeout: int = _env_int("DISCOVERY_ZERO_TACTIC_PROVER_TIMEOUT", 300)

    # ------------------------------------------------------------------ #
    # Experiment Sandbox                                                  #
    # ------------------------------------------------------------------ #
    experiment_backend: str = _env("DISCOVERY_ZERO_EXPERIMENT_BACKEND", "local")  # "local" | "docker" | "sandbox"
    experiment_memory_limit_mb: int = _env_int("DISCOVERY_ZERO_EXPERIMENT_MEMORY_MB", 512)
    experiment_cpu_limit: int = _env_int("DISCOVERY_ZERO_EXPERIMENT_CPU_LIMIT", 60)

    # ------------------------------------------------------------------ #
    # Lean Tactic Prover                                                  #
    # ------------------------------------------------------------------ #
    tactic_prover_enabled: bool = _env_bool("DISCOVERY_ZERO_TACTIC_PROVER_ENABLED", False)
    tactic_prover_max_depth: int = _env_int("DISCOVERY_ZERO_TACTIC_PROVER_MAX_DEPTH", 50)
    tactic_prover_beam_width: int = _env_int("DISCOVERY_ZERO_TACTIC_PROVER_BEAM_WIDTH", 8)
    tactic_prover_candidates_per_goal: int = _env_int(
        "DISCOVERY_ZERO_TACTIC_PROVER_CANDIDATES", 8
    )

    # ------------------------------------------------------------------ #
    # Token Budget                                                        #
    # ------------------------------------------------------------------ #
    token_budget_prompt: int = _env_int("DISCOVERY_ZERO_TOKEN_BUDGET_PROMPT", 0)
    token_budget_completion: int = _env_int("DISCOVERY_ZERO_TOKEN_BUDGET_COMPLETION", 0)

    # ------------------------------------------------------------------ #
    # DL / RL                                                             #
    # ------------------------------------------------------------------ #
    # Whether to apply a trained NeuralBPCorrector after each Gaia BP run.
    # Requires neural_bp_model_path to be set and PyTorch to be installed.
    neural_bp_enabled: bool = _env_bool("DISCOVERY_ZERO_NEURAL_BP_ENABLED", False)
    # Path to a trained NeuralBPCorrector checkpoint (.pt).
    neural_bp_model_path: str = _env("DISCOVERY_ZERO_NEURAL_BP_MODEL_PATH", "")
    # Correction strength in [0, 1]: 0 = pure Gaia BP, 1 = pure neural correction.
    neural_bp_correction_strength: float = _env_float("DISCOVERY_ZERO_NEURAL_BP_CORRECTION_STRENGTH", 0.3)

    # Whether to construct and (optionally) load a ProcessAdvantageVerifier.
    pav_enabled: bool = _env_bool("DISCOVERY_ZERO_PAV_ENABLED", False)
    # Path to a trained PAV checkpoint (.pt). If set and pav_enabled, loaded on start.
    pav_model_path: str = _env("DISCOVERY_ZERO_PAV_MODEL_PATH", "")

    expert_iter_enabled: bool = _env_bool("DISCOVERY_ZERO_EXPERT_ITER_ENABLED", False)

    # New iterative discovery stack
    enable_mcts: bool = _env_bool("DISCOVERY_ZERO_ENABLE_MCTS", False)
    enable_evolutionary_experiments: bool = _env_bool("DISCOVERY_ZERO_ENABLE_EVOLUTIONARY_EXPERIMENTS", True)
    enable_continuation_verification: bool = _env_bool("DISCOVERY_ZERO_ENABLE_CONTINUATION_VERIFICATION", True)
    enable_retrieval: bool = _env_bool(
        "DISCOVERY_ZERO_ENABLE_RETRIEVAL",
        bool(_env("EMBEDDING_API_BASE")),
    )
    enable_problem_variants: bool = _env_bool("DISCOVERY_ZERO_ENABLE_PROBLEM_VARIANTS", False)
    enable_analogy: bool = _env_bool("DISCOVERY_ZERO_ENABLE_ANALOGY", True)
    enable_specialize: bool = _env_bool("DISCOVERY_ZERO_ENABLE_SPECIALIZE", True)
    enable_decompose: bool = _env_bool("DISCOVERY_ZERO_ENABLE_DECOMPOSE", True)
    enable_claim_verifier: bool = _env_bool("DISCOVERY_ZERO_ENABLE_CLAIM_VERIFIER", True)
    # Dedicated model for claim verification code generation.
    # Should be a fast model with strong Python/math code ability (e.g. gpt-4o, gpt-4.1).
    # Defaults to empty (uses main model) if not set.
    claim_verification_model: str = _env("DISCOVERY_ZERO_CLAIM_VERIFICATION_MODEL", "")
    claim_verification_max_claims: int = _env_int("DISCOVERY_ZERO_CLAIM_VERIFICATION_MAX_CLAIMS", 3)
    claim_extraction_model: str = _env("DISCOVERY_ZERO_CLAIM_EXTRACTION_MODEL", "")
    max_claims_per_memo: int = _env_int("DISCOVERY_ZERO_MAX_CLAIMS_PER_MEMO", 10)
    verification_parallel_workers: int = _env_int("DISCOVERY_ZERO_VERIFICATION_PARALLEL_WORKERS", 3)
    verification_loop_enabled: bool = _env_bool("DISCOVERY_ZERO_VERIFICATION_LOOP_ENABLED", True)
    lean_feedback_enabled: bool = _env_bool("DISCOVERY_ZERO_LEAN_FEEDBACK_ENABLED", True)
    unverified_claim_prior: float = _env_float("DISCOVERY_ZERO_UNVERIFIED_CLAIM_PRIOR", 0.15)
    default_confidence_plausible: float = _env_float("DISCOVERY_ZERO_DEFAULT_CONFIDENCE_PLAUSIBLE", 0.5)
    default_confidence_experiment: float = _env_float("DISCOVERY_ZERO_DEFAULT_CONFIDENCE_EXPERIMENT", 0.85)
    default_confidence_lean: float = _env_float("DISCOVERY_ZERO_DEFAULT_CONFIDENCE_LEAN", 0.99)
    default_confidence_analogy: float = _env_float("DISCOVERY_ZERO_DEFAULT_CONFIDENCE_ANALOGY", 0.55)
    default_confidence_decompose: float = _env_float("DISCOVERY_ZERO_DEFAULT_CONFIDENCE_DECOMPOSE", 0.6)
    default_confidence_specialize: float = _env_float("DISCOVERY_ZERO_DEFAULT_CONFIDENCE_SPECIALIZE", 0.75)
    default_confidence_retrieve: float = _env_float("DISCOVERY_ZERO_DEFAULT_CONFIDENCE_RETRIEVE", 0.4)
    experiment_prior_cap: float = _env_float("DISCOVERY_ZERO_EXPERIMENT_PRIOR_CAP", 0.85)
    verified_prior_floor: float = _env_float("DISCOVERY_ZERO_VERIFIED_PRIOR_FLOOR", 0.6)
    inconclusive_prior_cap: float = _env_float("DISCOVERY_ZERO_INCONCLUSIVE_PRIOR_CAP", 0.4)
    refutation_prior_multiplier: float = _env_float("DISCOVERY_ZERO_REFUTATION_PRIOR_MULTIPLIER", 0.3)
    bridge_grade_prior_a: float = _env_float("DISCOVERY_ZERO_GRADE_PRIOR_A", 0.92)
    bridge_grade_prior_b: float = _env_float("DISCOVERY_ZERO_GRADE_PRIOR_B", 0.70)
    bridge_grade_prior_c: float = _env_float("DISCOVERY_ZERO_GRADE_PRIOR_C", 0.50)
    bridge_grade_prior_d: float = _env_float("DISCOVERY_ZERO_GRADE_PRIOR_D", 0.30)
    bridge_edge_confidence: float = _env_float("DISCOVERY_ZERO_BRIDGE_EDGE_CONFIDENCE", 0.85)
    bp_propagation_threshold: int = _env_int("DISCOVERY_ZERO_BP_PROPAGATION_THRESHOLD", 1)
    max_decompose_depth: int = _env_int("DISCOVERY_ZERO_MAX_DECOMPOSE_DEPTH", 4)
    structural_complexity_threshold: int = _env_int("DISCOVERY_ZERO_STRUCTURAL_COMPLEXITY_THRESHOLD", 2)
    enable_knowledge_retrieval: bool = _env_bool("DISCOVERY_ZERO_ENABLE_KNOWLEDGE_RETRIEVAL", True)

    mcts_max_iterations: int = _env_int("DISCOVERY_ZERO_MCTS_MAX_ITERATIONS", 50)
    mcts_max_time_seconds: float = _env_float("DISCOVERY_ZERO_MCTS_MAX_TIME_SECONDS", 14400.0)
    mcts_post_action_budget_seconds: float = _env_float("DISCOVERY_ZERO_MCTS_POST_ACTION_BUDGET", 300.0)
    mcts_c_puct: float = _env_float("DISCOVERY_ZERO_MCTS_C_PUCT", 1.4)
    mcts_num_simulations_per_expand: int = _env_int("DISCOVERY_ZERO_MCTS_NUM_SIMS", 3)
    mcts_progressive_widening_base: float = _env_float("DISCOVERY_ZERO_MCTS_PW_BASE", 1.5)
    mcts_specialization_threshold: int = _env_int("DISCOVERY_ZERO_MCTS_SPECIALIZATION_THRESHOLD", 3)
    mcts_replan_on_stuck: int = _env_int("DISCOVERY_ZERO_MCTS_REPLAN_ON_STUCK", 2)

    continuation_num_samples: int = _env_int("DISCOVERY_ZERO_CONTINUATION_NUM_SAMPLES", 4)
    continuation_consistency_threshold: float = _env_float("DISCOVERY_ZERO_CONTINUATION_THRESHOLD", 0.6)

    retrieval_max_results: int = _env_int("DISCOVERY_ZERO_RETRIEVAL_MAX_RESULTS", 6)
    retrieval_min_similarity: float = _env_float("DISCOVERY_ZERO_RETRIEVAL_MIN_SIMILARITY", 0.15)
    retrieval_graph_proximity_weight: float = _env_float("DISCOVERY_ZERO_RETRIEVAL_GRAPH_WEIGHT", 0.25)
    retrieval_use_gaia_storage: bool = _env_bool("DISCOVERY_ZERO_RETRIEVAL_USE_GAIA_STORAGE", False)
    retrieval_gaia_vector_top_k: int = _env_int("DISCOVERY_ZERO_RETRIEVAL_GAIA_TOP_K", 8)
    embedding_api_base: str = _env("EMBEDDING_API_BASE")
    gaia_api_base: str = _env("GAIA_API_BASE")

    external_prm_provider: str = _env("DISCOVERY_ZERO_EXTERNAL_PRM_PROVIDER", "chat_completion")
    external_prm_api_base: str = _env("EXTERNAL_PRM_API_BASE")
    external_prm_api_key: str = _env("EXTERNAL_PRM_API_KEY")
    external_prm_model: str = _env("EXTERNAL_PRM_MODEL")
    pav_blend_decay_experiences: int = _env_int("DISCOVERY_ZERO_PAV_BLEND_DECAY_EXPERIENCES", 500)
    expert_iter_checkpoint_dir: str = _env("DISCOVERY_ZERO_EXPERT_ITER_CHECKPOINT_DIR", "./expert_iter_checkpoints")

    # ------------------------------------------------------------------ #
    # Speculative Decomposition                                           #
    # ------------------------------------------------------------------ #
    spec_decomp_num_candidates: int = _env_int("DISCOVERY_ZERO_SPEC_DECOMP_CANDIDATES", 3)
    spec_decomp_enabled: bool = _env_bool("DISCOVERY_ZERO_SPEC_DECOMP_ENABLED", False)

    # ------------------------------------------------------------------ #
    # Paths                                                               #
    # ------------------------------------------------------------------ #
    @property
    def project_root(self) -> Path:
        return _get_project_root()

    @property
    def skills_dir(self) -> Path:
        return _get_project_root() / "skills"

    @property
    def lean_workspace(self) -> Path:
        custom = os.environ.get("DISCOVERY_ZERO_LEAN_WORKSPACE", "").strip()
        if custom:
            return Path(custom)
        return _get_project_root() / "lean_workspace"

    @property
    def default_graph_path(self) -> Path:
        return Path("./discovery_zero_graph.json")


# Module-level singleton — import this everywhere instead of constructing a new instance
CONFIG = ZeroConfig()
