#!/usr/bin/env python3
"""
Integration Test for Halyk Agent.
Verifies all components can be imported and basic functionality works.
"""
from __future__ import annotations
import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger


def test_imports():
    """Test all module imports."""
    logger.info("Testing imports...")

    try:
        # Config
        from halyk_agent.config import settings, get_settings
        logger.success("✓ config")

        # Models
        from halyk_agent.models import (
            DocumentMetadata, TextChunk, ExtractedTable, Transaction,
            Entity, Evidence, ReasoningStep, ProofBundle,
            CounterfactualResult, SubmissionRecord, SubmissionOutput
        )
        logger.success("✓ models")

        # Ingestion
        from halyk_agent.ingestion import (
            PDFIngestionPipeline, ParsedDocument, process_directory,
            TransactionLoader, load_transactions
        )
        logger.success("✓ ingestion")

        # Retrieval
        from halyk_agent.retrieval import (
            HybridRetriever, RetrievalResult, create_retriever, BGE_M3_Embedder
        )
        logger.success("✓ retrieval")

        # Graph
        from halyk_agent.graph import (
            KuzuGraph, EntityResolver, create_graph, create_resolver
        )
        logger.success("✓ graph")

        # Agents
        from halyk_agent.agents import (
            HalykAgent, AgentState, create_agent
        )
        logger.success("✓ agents")

        # Validation
        from halyk_agent.validation import (
            Z3Validator, CalculationEngine, VerificationResult
        )
        logger.success("✓ validation")

        # Proof
        from halyk_agent.proof import (
            SelfConsistencyVoter, CounterfactualAnalyzer, ProofBundleGenerator,
            create_voter, create_counterfactual_analyzer, create_proof_generator
        )
        logger.success("✓ proof")

        # Eval
        from halyk_agent.eval import (
            EvalHarness, EvalMetrics, create_harness
        )
        logger.success("✓ eval")

        # Main package
        import halyk_agent
        logger.success(f"✓ halyk_agent v{halyk_agent.__version__}")

        return True

    except Exception as e:
        logger.error(f"Import failed: {e}")
        traceback.print_exc()
        return False


def test_settings():
    """Test settings loading."""
    logger.info("Testing settings...")

    try:
        from halyk_agent.config import get_settings
        settings = get_settings()

        assert settings.app.name == "halyk-agent"
        assert settings.llm.model == "gpt-4o"
        assert settings.embedding.model == "BAAI/bge-m3"
        assert settings.vector_db.collection_name == "halyk_documents"
        assert settings.retrieval.top_k == 20

        logger.success("✓ settings loaded correctly")
        return True
    except Exception as e:
        logger.error(f"Settings test failed: {e}")
        traceback.print_exc()
        return False


def test_models():
    """Test model instantiation."""
    logger.info("Testing models...")

    try:
        from halyk_agent.models import (
            DocumentMetadata, TextChunk, ExtractedTable, Transaction,
            BoundingBox, Evidence, ReasoningStep, ProofBundle
        )
        from datetime import datetime
        from uuid import uuid4

        # BoundingBox
        bbox = BoundingBox(x0=100, y0=200, x1=300, y1=400, page=1)
        assert bbox.width == 200
        assert bbox.height == 200

        # DocumentMetadata
        meta = DocumentMetadata(
            doc_id=str(uuid4()),
            title="Test Doc",
            source_path="/test.pdf",
            page_count=5,
            file_hash="abc123",
        )
        assert meta.doc_type.value == "unknown"

        # Transaction
        txn = Transaction(
            txn_id="TXN_001",
            date=datetime.now(),
            amount=100000,
            sender="Org A",
            receiver="Org B",
        )
        assert txn.currency == "KZT"

        # Evidence
        ev = Evidence(
            claim="Test claim",
            source_doc_id="doc_1",
            source_type="text",
            page=1,
            confidence=0.9,
        )
        assert ev.verified == False

        logger.success("✓ models work correctly")
        return True

    except Exception as e:
        logger.error(f"Models test failed: {e}")
        traceback.print_exc()
        return False


def test_calculation_engine():
    """Test calculation engine."""
    logger.info("Testing calculation engine...")

    try:
        from halyk_agent.validation import CalculationEngine

        engine = CalculationEngine()

        # Simple arithmetic
        result = engine.evaluate("2 + 2 * 3", {})
        assert result == 8.0

        # With variables
        result = engine.evaluate("amount * rate", {"amount": 100000, "rate": 0.02})
        assert result == 2000.0

        # Formula evaluation
        result = engine.evaluate_formula("principal * (1 + rate) ** years", {
            "principal": 100000,
            "rate": 0.1,
            "years": 2,
        })
        assert result["success"] == True
        assert abs(result["value"] - 121000) < 1

        logger.success("✓ calculation engine works")
        return True

    except Exception as e:
        logger.error(f"Calculation test failed: {e}")
        traceback.print_exc()
        return False


def test_z3_validator():
    """Test Z3 validator."""
    logger.info("Testing Z3 validator...")

    try:
        from halyk_agent.validation import Z3Validator

        validator = Z3Validator()

        # Test commission rule
        calc = {
            "value": 1500,  # commission
            "inputs": {"amount": 100000},  # 1.5% - should pass
        }
        result = validator.verify_calculation("commission_calc", calc)
        assert result["all_valid"] == True

        # Test failing case
        calc_fail = {
            "value": 3000,  # 3% - should fail
            "inputs": {"amount": 100000},
        }
        result = validator.verify_calculation("commission_calc", calc_fail)
        assert result["all_valid"] == False

        logger.success("✓ Z3 validator works")
        return True

    except Exception as e:
        logger.error(f"Z3 test failed: {e}")
        traceback.print_exc()
        return False


def test_submission_schema():
    """Test submission.json schema."""
    logger.info("Testing submission schema...")

    try:
        from halyk_agent.models import SubmissionOutput, SubmissionRecord, ProofBundle, ReasoningStep
        from halyk_agent.proof import ProofBundleGenerator

        generator = ProofBundleGenerator()

        # Create minimal proof bundle
        from halyk_agent.models import Evidence
        from datetime import datetime
        from uuid import uuid4

        bundle = ProofBundle(
            query="Test query",
            decision="APPROVE",
            confidence=0.9,
            reasoning_trace=[
                ReasoningStep(
                    step_num=1,
                    claim="Test claim",
                    reasoning="Test reasoning",
                    confidence=0.9,
                    method="test",
                )
            ],
            evidence_bundle={"supporting_docs": ["doc_1"]},
        )

        # Export to submission.json
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        generator.to_submission_json([bundle], temp_path)

        # Verify it loads
        with open(temp_path, "r") as f:
            data = json.load(f)

        assert "submissions" in data
        assert len(data["submissions"]) == 1
        assert data["submissions"][0]["decision"] == "APPROVE"
        assert data["submissions"][0]["confidence"] == 0.9

        import os
        os.unlink(temp_path)

        logger.success("✓ submission schema works")
        return True

    except Exception as e:
        logger.error(f"Schema test failed: {e}")
        traceback.print_exc()
        return False


def test_cli_help():
    """Test CLI help."""
    logger.info("Testing CLI...")

    try:
        import subprocess
        import os
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).parent.parent / "src") + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, "-m", "halyk_agent.main", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            env=env,
        )
        logger.debug(f"CLI stdout: {result.stdout[:500]}")
        logger.debug(f"CLI stderr: {result.stderr[:500]}")
        assert result.returncode == 0, f"CLI failed with code {result.returncode}: {result.stderr}"
        assert "ingest" in result.stdout
        assert "run" in result.stdout
        assert "eval" in result.stdout

        logger.success("✓ CLI works")
        return True

    except Exception as e:
        logger.error(f"CLI test failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all integration tests."""
    logger.info("=" * 60)
    logger.info("HALYK AGENT INTEGRATION TESTS")
    logger.info("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Settings", test_settings),
        ("Models", test_models),
        ("Calculation Engine", test_calculation_engine),
        ("Z3 Validator", test_z3_validator),
        ("Submission Schema", test_submission_schema),
        ("CLI Help", test_cli_help),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        logger.info(f"\n--- {name} ---")
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"{name} crashed: {e}")
            failed += 1

    logger.info("\n" + "=" * 60)
    logger.info(f"RESULTS: {passed} passed, {failed} failed")
    logger.info("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import json
    success = run_all_tests()
    sys.exit(0 if success else 1)