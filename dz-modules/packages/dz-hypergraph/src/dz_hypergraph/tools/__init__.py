"""Tools layer: external integrations (LLM, Lean, policies)."""

from dz_hypergraph.tools.llm import (
    LLMConfig,
    LLMError,
    chat_completion,
    run_skill,
    extract_json_block,
    extract_text_content,
    load_skill_prompt,
)
from dz_hypergraph.tools.lean import verify_proof, decompose_proof_skeleton, get_workspace_path
from dz_hypergraph.tools.lean_policy import LeanBoundaryPolicy, LeanPolicyError, validate_lean_code
from dz_hypergraph.tools.external_prm import ExternalPRM, ExternalPRMConfig
from dz_hypergraph.tools.retrieval import HypergraphRetrievalIndex, RetrievalConfig, RetrievalResult
from dz_hypergraph.tools.experiment_templates import (
    ExperimentTemplate,
    TEMPLATES,
    get_template_catalog,
    render_template,
)
from dz_hypergraph.tools.verified_code_library import (
    VerifiedCodeLibrary,
    VerifiedFunction,
)
from dz_hypergraph.tools.gaia_client import (
    GaiaClient,
    GaiaClientConfig,
    build_gaia_client,
)
