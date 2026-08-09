"""
Halyk Agent - Main package.
"""
from halyk_agent.config import settings, get_settings
from halyk_agent.models import (
    DocumentMetadata,
    TextChunk,
    ExtractedTable,
    Transaction,
    Entity,
    Evidence,
    ReasoningStep,
    ProofBundle,
    CounterfactualResult,
    SubmissionRecord,
    SubmissionOutput,
)
from halyk_agent.ingestion import (
    PDFIngestionPipeline,
    ParsedDocument,
    process_directory,
    TransactionLoader,
    load_transactions,
)
from halyk_agent.retrieval import (
    HybridRetriever,
    RetrievalResult,
    create_retriever,
    BGE_M3_Embedder,
)
from halyk_agent.graph import (
    KuzuGraph,
    EntityResolver,
    create_graph,
    create_resolver,
)
from halyk_agent.agents import (
    HalykAgent,
    AgentState,
    create_agent,
)
from halyk_agent.validation import (
    Z3Validator,
    CalculationEngine,
    VerificationResult,
)
from halyk_agent.proof import (
    SelfConsistencyVoter,
    CounterfactualAnalyzer,
    ProofBundleGenerator,
    create_voter,
    create_counterfactual_analyzer,
    create_proof_generator,
)
from halyk_agent.eval import (
    EvalHarness,
    EvalMetrics,
)

__version__ = "0.1.0"

__all__ = [
    # Config
    "settings",
    "get_settings",
    # Models
    "DocumentMetadata",
    "TextChunk",
    "ExtractedTable",
    "Transaction",
    "Entity",
    "Evidence",
    "ReasoningStep",
    "ProofBundle",
    "CounterfactualResult",
    "SubmissionRecord",
    "SubmissionOutput",
    # Ingestion
    "PDFIngestionPipeline",
    "ParsedDocument",
    "process_directory",
    "TransactionLoader",
    "load_transactions",
    # Retrieval
    "HybridRetriever",
    "RetrievalResult",
    "create_retriever",
    "BGE_M3_Embedder",
    # Graph
    "KuzuGraph",
    "EntityResolver",
    "create_graph",
    "create_resolver",
    # Agents
    "HalykAgent",
    "AgentState",
    "create_agent",
    # Validation
    "Z3Validator",
    "CalculationEngine",
    "VerificationResult",
    # Proof
    "SelfConsistencyVoter",
    "CounterfactualAnalyzer",
    "ProofBundleGenerator",
    "create_voter",
    "create_counterfactual_analyzer",
    "create_proof_generator",
    # Eval
    "EvalHarness",
    "EvalMetrics",
]