# Halyk AI Challenge Agent

> **Production-ready agent system for the Halyk AI Challenge** — Document intelligence, temporal reasoning, financial calculation, and audit-grade proof generation.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HALYK AGENT PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│  │ INGEST   │───▶│ RETRIEVE │───▶│ RESOLVE  │───▶│ CALCULATE│             │
│  │          │    │          │    │          │    │          │             │
│  │ marker-  │    │ bge-m3   │    │ KuzuDB   │    │ numexpr  │             │
│  │ pdf +    │    │ dense +  │    │ Entity   │    │ + Z3     │             │
│  │ docling  │    │ sparse + │    │ Resolution│   │ Constraints│            │
│  │ Tables+  │    │ colbert  │    │ Bi-temp  │    │ Proof    │             │
│  │ BBox     │    │ + BM25   │    │ Graph    │    │          │             │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘             │
│       │                                        │                           │
│       ▼                                        ▼                           │
│  ┌──────────────────────────────────────────────────────────┐             │
│  │              VERIFIER (Z3 SMT Solver)                    │             │
│  │  • max_commission_rate  • min_reserve_ratio              │             │
│  │  • max_loan_to_value    • interest_rate_cap              │             │
│  │  • kyc_required         • aml_threshold                  │             │
│  │  • transaction_limit    • currency_control               │             │
│  └──────────────────────────────────────────────────────────┘             │
│       │                                        │                           │
│       ▼                                        ▼                           │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│  │ SYNTHESIZE│◀──│ VALIDATE │◀──│ COUNTER- │◀──│  VOTING  │             │
│  │          │    │          │    │ FACTUAL  │    │ (N=5)    │             │
│  │ Proof    │    │ Evidence │    │ Remove   │    │ Majority │             │
│  │ Bundles  │    │ Check    │    │ Docs/Txn │    │ Consensus│             │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## ✨ Key Features

| Feature | Technology | Why It Wins |
|---------|------------|-------------|
| **PDF Parsing** | `marker-pdf` + `docling` | SOTA table extraction with pixel-perfect bbox |
| **Embeddings** | `bge-m3` (dense+sparse+colbert) | Multilingual (kz/ru/en), hybrid search native |
| **Retrieval** | Qdrant + BM25 + Temporal filtering | Self-query by document validity period |
| **Entity Resolution** | KuzuDB + RapidFuzz + LLM verification | Bi-temporal graph (doc time + txn time) |
| **Calculation** | `numexpr` + `simpleeval` | Safe, fast, auditable math |
| **Business Rules** | Z3 SMT Solver | **Proves** compliance, doesn't just check |
| **Self-Consistency** | N=5 voting | Reduces hallucination, measures agreement |
| **Counterfactual** | Remove docs/txns, re-evaluate | Quantifies evidence criticality |
| **Proof Bundles** | Page + bbox + method citations | Audit-ready, judge-friendly output |

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker (recommended) or local CUDA for embeddings
- 16GB+ RAM, 8GB+ VRAM for bge-m3

### Option 1: Docker (Recommended)

```bash
# Build
docker build -t halyk-agent .

# Run ingestion
docker run --rm -v $(pwd)/data:/app/data halyk-agent \
  ingest --input-dir /app/data/raw --output-dir /app/data/processed

# Run agent
docker run --rm -v $(pwd)/data:/app/data halyk-agent \
  run --query "Should transaction TXN_123 be approved?" \
      --case-id CASE_001 \
      --transaction-date 2024-01-15 \
      --output /app/data/submission.json
```

### Option 2: Local Development

```bash
# Create venv
python3.11 -m venv .venv
source .venv/bin/activate

# Install
pip install -e ".[dev]"

# Download models (first run)
python -c "from flag_embedding import FlagModel; FlagModel('BAAI/bge-m3')"

# Run ingestion
python -m halyk_agent.main ingest \
  --input-dir data/raw \
  --output-dir data/processed \
  --transactions data/transactions.csv

# Run agent
python -m halyk_agent.main run \
  --query "Should transaction TXN_123 be approved?" \
  --case-id CASE_001 \
  --transaction-date 2024-01-15 \
  --votes 5 \
  --output submission.json
```

## 📁 Project Structure

```
halyk-agent/
├── config/
│   └── settings.yaml          # All configuration
├── data/
│   ├── raw/                   # Input PDFs, transaction CSV
│   └── processed/             # Parsed chunks, tables, embeddings
├── scripts/
│   └── run_pipeline.py        # End-to-end orchestration
├── src/halyk_agent/
│   ├── config/                # Settings management
│   ├── models/                # Pydantic data models
│   ├── ingestion/             # PDF + transaction parsing
│   ├── retrieval/             # Hybrid search (bge-m3 + BM25)
│   ├── graph/                 # KuzuDB entity resolution
│   ├── agents/                # LangGraph pipeline
│   ├── validation/            # Z3 constraints + calc engine
│   ├── proof/                 # Voting + counterfactual + bundles
│   ├── eval/                  # Evaluation harness
│   └── main.py                # CLI entry point
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   ├── test_graph.py
│   ├── test_validation.py
│   └── eval_harness.py
├── Dockerfile
├── pyproject.toml
└── README.md
```

## 🔧 Configuration

All settings in `config/settings.yaml` (override via environment variables):

```yaml
llm:
  provider: "openai"
  model: "gpt-4o"
  temperature: 0.1

embedding:
  model: "BAAI/bge-m3"
  device: "cuda"
  use_dense: true
  use_sparse: true
  use_colbert: true

vector_db:
  dense_weight: 0.5
  sparse_weight: 0.3
  colbert_weight: 0.2

retrieval:
  top_k: 20
  consistency_votes: 5
  counterfactual_samples: 3

validation:
  consistency_threshold: 0.7
  counterfactual_threshold: 0.15
```

## 📊 CLI Commands

```bash
# Ingest documents
halyk-agent ingest --input-dir data/raw --output-dir data/processed --transactions data/txns.csv

# Test retrieval
halyk-agent retrieve --query "комиссия 2023" --top-k 10

# Run full pipeline with voting
halyk-agent run --query "Should TXN_123 be approved?" \
    --case-id CASE_001 \
    --transaction-date 2024-01-15 \
    --votes 5 \
    --output submission.json

# Run evaluation
halyk-agent eval --test-cases data/test.json --golden data/golden.json --output-dir results/
```

## 📋 Output: submission.json

```json
{
  "submissions": [
    {
      "case_id": "CASE_001",
      "decision": "APPROVE",
      "confidence": 0.91,
      "reasoning": "Step 1: Retrieved 15 relevant chunks...\nStep 2: Resolved 8 entities...\nStep 3: Calculated commission = 45,000 KZT...\nStep 4: Business rule verification PASSED",
      "evidence": [
        {
          "source": "doc_abc123",
          "type": "table",
          "page": 4,
          "bbox": {"x0": 120, "y0": 450, "x1": 300, "y1": 480, "page": 4},
          "claim": "Commission rate: 1.5%",
          "confidence": 0.95
        }
      ]
    }
  ],
  "metadata": {
    "generated_at": "2024-01-15T10:30:00Z",
    "total_cases": 1
  }
}
```

## 🧪 Evaluation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Accuracy | > 0.90 | Decision correctness |
| Citation Quality | > 0.85 | Every claim has page+bbox+method |
| Counterfactual Robustness | > 0.80 | Decision stable under evidence removal |
| Schema Valid | 100% | Matches Submission.json exactly |

## 🔬 Advanced Usage

### Custom Business Rules (Z3)

```python
from halyk_agent.validation import Z3Validator

validator = Z3Validator()
# Add custom rule
def custom_rule(calc):
    s = Solver()
    s.add(Real("my_metric") <= 100)
    return validator._check_solver(s, "custom", {"my_metric": Real("my_metric")})

validator.rules["custom_rule"] = custom_rule
```

### Extend Entity Types

```python
# In config/settings.yaml
graph_db:
  entity_types:
    - "Organization"
    - "Person"
    - "Contract"
    - "Transaction"
    - "Document"
    - "Regulation"
    - "Counterparty"      # NEW
    - "BeneficialOwner"   # NEW
```

### Custom Retrieval Weights

```python
retriever = HybridRetriever()
# Override fusion weights
results = retriever.retrieve(query, top_k=10)
# Results include dense_score, sparse_score, colbert_score, bm25_score
```

## 📈 Performance Optimization

- **Embeddings**: Cache bge-m3 embeddings in Qdrant (persistent)
- **Graph**: KuzuDB is embedded, zero-latency for local queries
- **Voting**: Run N votes in parallel (ThreadPoolExecutor)
- **Token Usage**: Use `instructor` for structured output, reduce retries

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `marker-pdf` import error | `pip install marker-pdf[full]` |
| `docling` OCR fails | Install `tesseract-ocr-kaz` system package |
| CUDA OOM | Reduce `embedding.batch_size` to 8 |
| Qdrant connection refused | `docker run -d -p 6333:6333 qdrant/qdrant` |
| Z3 timeout | Increase `validation.z3_timeout_ms` |

## 📝 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Run tests: `pytest tests/ -v`
4. Submit PR with description of changes

---

**Built for Halyk AI Challenge 2024** — *Winning through audit-grade reasoning, not just accuracy.*