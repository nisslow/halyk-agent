"""
Configuration management for Halyk Agent.
Loads settings from YAML with environment variable overrides.
"""
from pathlib import Path
from typing import Any, Optional
from functools import lru_cache

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseModel):
    name: str = "halyk-agent"
    version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"


class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 8192
    timeout: int = 120
    max_retries: int = 3


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-m3"
    device: str = "cuda"
    batch_size: int = 32
    max_length: int = 8192
    use_dense: bool = True
    use_sparse: bool = True
    use_colbert: bool = True


class PDFParsingConfig(BaseModel):
    class MarkerConfig(BaseModel):
        use_llm: bool = False
        output_format: str = "markdown"
        extract_tables: bool = True
        extract_images: bool = False
        languages: list[str] = ["kz", "ru", "en"]

    class DoclingConfig(BaseModel):
        extract_tables: bool = True
        extract_figures: bool = False
        ocr_engine: str = "easyocr"
        languages: list[str] = ["kz", "ru", "en"]

    marker: MarkerConfig = Field(default_factory=MarkerConfig)
    docling: DoclingConfig = Field(default_factory=DoclingConfig)


class VectorDBConfig(BaseModel):
    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    collection_name: str = "halyk_documents"
    vector_size: int = 1024
    distance: str = "Cosine"
    dense_weight: float = 0.5
    sparse_weight: float = 0.3
    colbert_weight: float = 0.2
    enable_temporal_filter: bool = True


class GraphDBConfig(BaseModel):
    path: str = "./data/kuzu_db"
    entity_types: list[str] = [
        "Entity", "Organization", "Person", "Contract", "Regulation"
    ]
    relation_types: list[str] = [
        "APPEARS_IN", "HAS_TXN", "SUPERSEDES", "AMENDS",
        "APPENDIX_OF", "ISSUED_BY", "GOVERNS"
    ]


class IngestionConfig(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 200

    class TableExtractionConfig(BaseModel):
        min_rows: int = 2
        min_cols: int = 2
        header_detection: bool = True
        bbox_inclusion: bool = True

    class MetadataExtractionConfig(BaseModel):
        use_llm: bool = True
        extract_dates: bool = True
        extract_entities: bool = True
        extract_keywords: bool = True

    table_extraction: TableExtractionConfig = Field(default_factory=TableExtractionConfig)
    metadata_extraction: MetadataExtractionConfig = Field(default_factory=MetadataExtractionConfig)


class RetrievalConfig(BaseModel):
    top_k: int = 20
    consistency_votes: int = 5
    counterfactual_samples: int = 3

    class TemporalFilterConfig(BaseModel):
        enabled: bool = True
        transaction_date_field: str = "transaction_date"
        doc_valid_from_field: str = "valid_from"
        doc_valid_to_field: str = "valid_to"

    temporal_filter: TemporalFilterConfig = Field(default_factory=TemporalFilterConfig)


class ValidationConfig(BaseModel):
    z3_timeout_ms: int = 5000
    consistency_threshold: float = 0.7
    counterfactual_threshold: float = 0.15
    min_citations_per_claim: int = 1
    max_citations_per_claim: int = 5


class PipelineConfig(BaseModel):
    max_iterations: int = 3
    max_retries_per_node: int = 2

    class NodeTimeouts(BaseModel):
        planner: int = 30
        retriever: int = 60
        resolver: int = 30
        calculator: int = 30
        verifier: int = 30
        synthesizer: int = 30
        validator: int = 30

    node_timeouts: NodeTimeouts = Field(default_factory=NodeTimeouts)


class OutputConfig(BaseModel):
    proof_bundle_format: str = "json"
    include_bbox: bool = True
    include_extraction_method: bool = True
    include_confidence: bool = True
    submission_file: str = "submission.json"


class EvalConfig(BaseModel):
    harness_path: str = "./tests/eval_harness.py"
    test_dataset: str = "./data/test/"
    golden_file: str = "./data/golden.json"
    metrics: list[str] = [
        "accuracy", "precision", "recall", "f1",
        "citation_quality", "counterfactual_robustness"
    ]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        nested_model_default_partial_update=True,
    )

    app: AppConfig = Field(default_factory=AppConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    pdf_parsing: PDFParsingConfig = Field(default_factory=PDFParsingConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    graph_db: GraphDBConfig = Field(default_factory=GraphDBConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)


@lru_cache
def get_settings(config_path: Optional[Path] = None) -> Settings:
    """Load settings from YAML config file with env var overrides."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"

    yaml_data = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

    # Flatten nested dict for pydantic-settings
    flat_data = _flatten_dict(yaml_data)
    return Settings(**flat_data)


def _flatten_dict(d: dict, parent_key: str = "", sep: str = "_") -> dict:
    """Flatten nested dict for pydantic-settings."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# Global settings instance
settings = get_settings()