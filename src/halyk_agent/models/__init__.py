"""
Core data models for Halyk Agent.
Defines all types used across the pipeline.
"""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ConfigDict


class DocumentType(str, Enum):
    """Types of documents in the dataset."""
    REGULATION = "regulation"
    CONTRACT = "contract"
    FINANCIAL_REPORT = "financial_report"
    TAX_RETURN = "tax_return"
    BANK_STATEMENT = "bank_statement"
    INVOICE = "invoice"
    CORRESPONDENCE = "correspondence"
    INTERNAL_MEMO = "internal_memo"
    LEGAL_FILING = "legal_filing"
    UNKNOWN = "unknown"


class ExtractionMethod(str, Enum):
    """Method used to extract content."""
    MARKER_TEXT = "marker_text"
    MARKER_TABLE = "marker_table"
    DOCLING_TEXT = "docling_text"
    DOCLING_TABLE = "docling_table"
    OCR = "ocr"
    MANUAL = "manual"


class BoundingBox(BaseModel):
    """Bounding box in PDF coordinates (points, origin bottom-left)."""
    model_config = ConfigDict(frozen=True)
    x0: float
    y0: float
    x1: float
    y1: float
    page: int

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def to_dict(self) -> dict[str, float | int]:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1, "page": self.page}


class DocumentMetadata(BaseModel):
    """Metadata extracted from a document."""
    doc_id: str
    title: Optional[str] = None
    doc_type: DocumentType = DocumentType.UNKNOWN
    language: str = "ru"
    # Temporal fields
    created_date: Optional[datetime] = None
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    version: Optional[str] = None
    supersedes: Optional[str] = None  # doc_id of superseded document
    amends: Optional[str] = None
    appendix_of: Optional[str] = None
    # Entities
    organizations: list[str] = Field(default_factory=list)
    persons: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    # Source
    source_path: str
    page_count: int
    file_hash: str
    processed_at: datetime = Field(default_factory=datetime.utcnow)


class TableCell(BaseModel):
    """Single cell in an extracted table."""
    model_config = ConfigDict(frozen=True)
    row: int
    col: int
    text: str
    bbox: Optional[BoundingBox] = None
    is_header: bool = False
    colspan: int = 1
    rowspan: int = 1


class ExtractedTable(BaseModel):
    """Extracted table with full provenance."""
    table_id: str = Field(default_factory=lambda: str(uuid4()))
    page: int
    bbox: BoundingBox
    headers: list[str] = Field(default_factory=list)
    rows: list[list[TableCell]] = Field(default_factory=list)
    extraction_method: ExtractionMethod
    confidence: float = 1.0
    # Metadata
    caption: Optional[str] = None
    section_header: Optional[str] = None


class TextChunk(BaseModel):
    """Text chunk with provenance."""
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    doc_id: str
    text: str
    page: int
    bbox: Optional[BoundingBox] = None
    section_header: Optional[str] = None
    extraction_method: ExtractionMethod
    metadata: DocumentMetadata
    # Embeddings (populated later)
    dense_embedding: Optional[list[float]] = None
    sparse_embedding: Optional[dict[int, float]] = None  # {token_id: weight}
    colbert_embedding: Optional[list[list[float]]] = None


class Transaction(BaseModel):
    """Transaction from registry."""
    txn_id: str
    date: datetime
    amount: float
    currency: str = "KZT"
    sender: str
    receiver: str
    sender_bin: Optional[str] = None  # Business Identification Number
    receiver_bin: Optional[str] = None
    purpose: Optional[str] = None
    category: Optional[str] = None
    raw_data: dict[str, Any] = Field(default_factory=dict)


class Entity(BaseModel):
    """Resolved entity in the knowledge graph."""
    entity_id: str = Field(default_factory=lambda: str(uuid4()))
    canonical_name: str
    entity_type: str  # Organization, Person, Contract, etc.
    aliases: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    # Provenance
    source_docs: list[str] = Field(default_factory=list)
    source_txns: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Evidence(BaseModel):
    """Evidence citation for a claim."""
    claim_id: str = Field(default_factory=lambda: str(uuid4()))
    claim: str
    source_doc_id: str
    source_type: Literal["text", "table", "transaction", "calculation"]
    # For text/table
    page: Optional[int] = None
    bbox: Optional[BoundingBox] = None
    chunk_id: Optional[str] = None
    table_id: Optional[str] = None
    row_idx: Optional[int] = None
    col_idx: Optional[int] = None
    # For transaction
    txn_id: Optional[str] = None
    # For calculation
    formula: Optional[str] = None
    inputs: list[str] = Field(default_factory=list)
    # Metadata
    extraction_method: Optional[ExtractionMethod] = None
    confidence: float = 1.0
    verified: bool = False


class ReasoningStep(BaseModel):
    """Single step in reasoning trace."""
    step_id: str = Field(default_factory=lambda: str(uuid4()))
    step_num: int
    claim: str
    reasoning: str
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = 1.0
    method: str  # retrieval, calculation, resolution, verification


class CounterfactualResult(BaseModel):
    """Result of counterfactual analysis."""
    removed_element: str  # doc_id, table_id, or txn_id
    original_decision: str
    counterfactual_decision: str
    original_confidence: float
    counterfactual_confidence: float
    decision_flipped: bool
    confidence_delta: float


class ProofBundle(BaseModel):
    """Complete proof bundle for a decision."""
    decision_id: str = Field(default_factory=lambda: str(uuid4()))
    query: str
    decision: str
    confidence: float
    reasoning_trace: list[ReasoningStep] = Field(default_factory=list)
    evidence_bundle: dict[str, Any] = Field(default_factory=dict)
    counterfactual_analysis: list[CounterfactualResult] = Field(default_factory=list)
    business_rule_validation: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SubmissionRecord(BaseModel):
    """Single record in submission.json."""
    case_id: str
    decision: str
    confidence: float
    reasoning: str
    evidence: list[dict[str, Any]] = Field(default_factory=list)


class SubmissionOutput(BaseModel):
    """Full submission.json structure."""
    submissions: list[SubmissionRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)