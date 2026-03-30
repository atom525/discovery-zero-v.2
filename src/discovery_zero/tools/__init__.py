"""Tools layer: external integrations (LLM, Lean, policies)."""

from discovery_zero.tools.llm import (
    LLMConfig,
    LLMError,
    chat_completion,
    run_skill,
    extract_json_block,
    extract_text_content,
    load_skill_prompt,
)
from discovery_zero.tools.lean import verify_proof, decompose_proof_skeleton, get_workspace_path
from discovery_zero.tools.lean_policy import LeanBoundaryPolicy, LeanPolicyError, validate_lean_code
from discovery_zero.tools.external_prm import ExternalPRM, ExternalPRMConfig
from discovery_zero.tools.retrieval import HypergraphRetrievalIndex, RetrievalConfig, RetrievalResult
from discovery_zero.tools.experiment_templates import (
    ExperimentTemplate,
    TEMPLATES,
    get_template_catalog,
    render_template,
)
from discovery_zero.tools.verified_code_library import (
    VerifiedCodeLibrary,
    VerifiedFunction,
)
from discovery_zero.tools.gaia_client import (
    GaiaClient,
    GaiaClientConfig,
    build_gaia_client,
)
