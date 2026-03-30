"""Planning layer: bridge plans, orchestration, proof search."""

from discovery_zero.planning.bridge import (
    BridgePlan,
    BridgeProposition,
    BridgeReasoningStep,
    BridgeValidationError,
    validate_bridge_plan_payload,
    materialize_bridge_nodes,
)
from discovery_zero.planning.orchestrator import (
    execute_action,
    ingest_action_output,
    run_loop,
    ActionResult,
    OrchestrationError,
)
from discovery_zero.planning.htps import HTPSState, htps_step
from discovery_zero.planning.continuation_verifier import (
    ContinuationConfig,
    ContinuationVerifier,
    StepVerificationResult,
)
from discovery_zero.planning.experiment_evolution import (
    EvolutionConfig,
    ExperimentEvolver,
    ExperimentProgram,
)
from discovery_zero.planning.mcts_engine import (
    MCTSConfig,
    MCTSDiscoveryEngine,
    MCTSDiscoveryResult,
)
from discovery_zero.planning.verification_loop import (
    VerificationLoop,
    VerificationLoopConfig,
    VerificationLoopResult,
    VerificationIterationTrace,
)
from discovery_zero.planning.problem_variants import (
    ProblemVariant,
    ProblemVariantGenerator,
)
from discovery_zero.planning.claim_verifier import (
    ClaimVerifier,
    ClaimVerificationResult,
    VerifiableClaim,
)
from discovery_zero.planning.claim_pipeline import (
    ClaimPipeline,
    ClaimPipelineConfig,
)
from discovery_zero.planning.analogy import (
    Analogy,
    AnalogyEngine,
    TransferResult,
)
from discovery_zero.planning.specialize import (
    Pattern,
    Specialization,
    SpecializeEngine,
)
from discovery_zero.planning.decompose import (
    DecomposeEngine,
    FormalSubGoal,
    SubProblem,
)
from discovery_zero.planning.lean_feedback import (
    LeanGapAnalysis,
    LeanFeedbackParser,
    StructuralClaimRouter,
    StructuralVerificationPlan,
)
from discovery_zero.planning.knowledge_retrieval import (
    KnowledgeFact,
    KnowledgeRetriever,
)
from discovery_zero.planning.incremental_codegen import (
    IncrementalCodeGenerator,
    IncrementalCodegenResult,
)
