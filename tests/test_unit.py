# Halyk Agent Tests

# Unit tests
def test_bounding_box():
    from halyk_agent.models import BoundingBox
    bbox = BoundingBox(x0=0, y0=0, x1=100, y1=100, page=1)
    assert bbox.width == 100
    assert bbox.height == 100

def test_document_metadata():
    from halyk_agent.models import DocumentMetadata, DocumentType
    from uuid import uuid4
    meta = DocumentMetadata(
        doc_id=str(uuid4()),
        title="Test",
        source_path="/test.pdf",
        page_count=1,
        file_hash="abc",
        doc_type=DocumentType.REGULATION,
    )
    assert meta.doc_type == DocumentType.REGULATION

def test_transaction():
    from halyk_agent.models import Transaction
    from datetime import datetime
    txn = Transaction(
        txn_id="TXN_001",
        date=datetime.now(),
        amount=100000,
        sender="A",
        receiver="B",
    )
    assert txn.amount == 100000
    assert txn.currency == "KZT"

def test_calculation_engine():
    from halyk_agent.validation import CalculationEngine
    engine = CalculationEngine()
    assert engine.evaluate("1 + 1", {}) == 2.0
    assert engine.evaluate("a * b", {"a": 5, "b": 3}) == 15.0

def test_z3_validator_commission():
    from halyk_agent.validation import Z3Validator
    validator = Z3Validator()
    
    # Should pass: 1.5% commission
    result = validator.verify_calculation("commission", {
        "value": 1500,
        "inputs": {"amount": 100000},
    })
    assert result["all_valid"] == True
    
    # Should fail: 3% commission
    result = validator.verify_calculation("commission", {
        "value": 3000,
        "inputs": {"amount": 100000},
    })
    assert result["all_valid"] == False

def test_z3_validator_kyc():
    from halyk_agent.validation import Z3Validator
    validator = Z3Validator()
    
    # Should pass: amount < 500000, no KYC needed
    result = validator.verify_calculation("kyc", {
        "inputs": {"amount": 100000, "kyc_completed": False},
    })
    assert result["all_valid"] == True
    
    # Should fail: amount > 500000, KYC not done
    result = validator.verify_calculation("kyc", {
        "inputs": {"amount": 1000000, "kyc_completed": False},
    })
    assert result["all_valid"] == False

def test_submission_output():
    from halyk_agent.models import SubmissionOutput, SubmissionRecord
    from halyk_agent.proof import ProofBundleGenerator
    from halyk_agent.models import ProofBundle, ReasoningStep
    import tempfile
    import json
    import os
    
    generator = ProofBundleGenerator()
    bundle = ProofBundle(
        query="Test",
        decision="APPROVE",
        confidence=0.9,
        reasoning_trace=[
            ReasoningStep(step_num=1, claim="Test", reasoning="Test", confidence=0.9, method="test")
        ],
    )
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name
    
    generator.to_submission_json([bundle], path)
    
    with open(path) as f:
        data = json.load(f)
    
    assert data["submissions"][0]["decision"] == "APPROVE"
    os.unlink(path)

def test_settings_load():
    from halyk_agent.config import get_settings
    settings = get_settings()
    assert settings.app.name == "halyk-agent"
    assert settings.embedding.model == "BAAI/bge-m3"