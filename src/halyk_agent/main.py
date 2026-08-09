#!/usr/bin/env python3
"""
Halyk AI Challenge Agent - Main Entry Point.

Usage:
    python -m halyk_agent.main ingest --input-dir data/raw --output-dir data/processed
    python -m halyk_agent.main retrieve --query "commission rate 2023" --top-k 10
    python -m halyk_agent.main resolve --documents data/processed --transactions data/transactions.parquet
    python -m halyk_agent.main run --query "Should transaction TXN_123 be approved?" --case-id CASE_001 --transaction-date 2024-01-15
    python -m halyk_agent.main eval --test-cases data/test_cases.json --golden data/golden.json --output-dir results/
"""
from __future__ import annotations
import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from halyk_agent.config import settings
from halyk_agent.ingestion import PDFIngestionPipeline, TransactionLoader, process_directory
from halyk_agent.retrieval import create_retriever
from halyk_agent.graph import create_graph, create_resolver
from halyk_agent.agents import create_agent
from halyk_agent.proof import create_voter, create_proof_generator
from halyk_agent.eval import create_harness


def setup_logging(debug: bool = False):
    """Configure logging."""
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if debug else "INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )


def cmd_ingest(args):
    """Ingest PDF documents and transaction registry."""
    logger.info("Starting ingestion pipeline...")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return 1

    # Process PDFs
    logger.info(f"Processing PDFs from {input_dir}")
    parsed_docs = process_directory(input_dir, output_dir)

    # Process transactions if provided
    if args.transactions:
        txn_path = Path(args.transactions)
        if txn_path.exists():
            logger.info(f"Loading transactions from {txn_path}")
            loader = TransactionLoader()
            transactions = loader.load(txn_path)
            # Save as parquet for fast loading
            loader.save_parquet(transactions, output_dir / "transactions.parquet")
        else:
            logger.warning(f"Transaction file not found: {txn_path}")

    logger.info(f"Ingestion complete. Processed {len(parsed_docs)} documents.")
    return 0


def cmd_retrieve(args):
    """Test retrieval."""
    logger.info("Testing retrieval...")

    retriever = create_retriever()

    # Load existing index or build from processed data
    # For testing, we'll just run a query
    results = retriever.retrieve(
        query=args.query,
        top_k=args.top_k,
    )

    for i, result in enumerate(results):
        print(f"\n--- Result {i+1} (score: {result.score:.4f}) ---")
        print(f"Doc: {result.chunk.doc_id}")
        print(f"Page: {result.chunk.page}")
        print(f"Method: {result.chunk.extraction_method.value}")
        print(f"Text: {result.chunk.text[:200]}...")

    return 0


def cmd_resolve(args):
    """Run entity resolution."""
    logger.info("Running entity resolution...")

    # Load processed documents
    processed_dir = Path(args.documents)
    # This would load the parsed documents and run resolution
    logger.info("Entity resolution not fully implemented in CLI yet")
    return 0


def cmd_run(args):
    """Run the full agent pipeline on a query."""
    logger.info(f"Running agent for query: {args.query}")

    agent = create_agent()

    # Parse transaction date if provided
    txn_date = None
    if args.transaction_date:
        txn_date = datetime.fromisoformat(args.transaction_date)

    # Run with self-consistency voting
    voter = create_voter(agent, n_votes=args.votes)

    bundle = voter.vote(
        query=args.query,
        case_id=args.case_id,
        transaction_date=txn_date,
    )

    # Generate proof bundle
    generator = create_proof_generator()

    # Output results
    print(f"\n{'='*60}")
    print(f"DECISION: {bundle.decision}")
    print(f"CONFIDENCE: {bundle.confidence:.4f}")
    print(f"{'='*60}")

    print("\nREASONING TRACE:")
    for step in bundle.reasoning_trace:
        safe_claim = step.claim.replace("\u2192", "->") if step.claim else ""
        print(f"  Step {step.step_num}: {safe_claim}")
        print(f"    Confidence: {step.confidence:.2f}")
        print(f"    Evidence: {len(step.evidence)} citations")

    print("\nCOUNTERFACTUAL ANALYSIS:")
    for cf in bundle.counterfactual_analysis:
        status = "FLIPPED" if cf.decision_flipped else "STABLE"
        print(f"  Remove {cf.removed_element}: {cf.original_decision} -> {cf.counterfactual_decision} [{status}]")

    # Save submission.json
    if args.output:
        output_path = Path(args.output)
        generator.to_submission_json([bundle], str(output_path))
        logger.info(f"Submission saved to {output_path}")

    return 0


def cmd_eval(args):
    """Run evaluation harness."""
    logger.info("Running evaluation...")

    agent = create_agent()
    harness = create_harness(
        agent=agent,
        golden_path=Path(args.golden),
        test_cases_path=Path(args.test_cases),
    )

    output_dir = Path(args.output_dir) if args.output_dir else None
    metrics = harness.run_evaluation(output_dir)

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy:              {metrics.accuracy:.4f}")
    print(f"Precision:             {metrics.precision:.4f}")
    print(f"Recall:                {metrics.recall:.4f}")
    print(f"F1 Score:              {metrics.f1:.4f}")
    print(f"Citation Quality:      {metrics.citation_quality:.4f}")
    print(f"Counterfactual Robust: {metrics.counterfactual_robustness:.4f}")
    print(f"Schema Valid:          {metrics.schema_valid}")
    print(f"{'='*60}")

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Halyk AI Challenge Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents and transactions")
    ingest_parser.add_argument("--input-dir", required=True, help="Directory with PDF files")
    ingest_parser.add_argument("--output-dir", required=True, help="Output directory for processed data")
    ingest_parser.add_argument("--transactions", help="Path to transaction registry file")

    # Retrieve command
    retrieve_parser = subparsers.add_parser("retrieve", help="Test retrieval")
    retrieve_parser.add_argument("--query", required=True, help="Search query")
    retrieve_parser.add_argument("--top-k", type=int, default=10, help="Number of results")

    # Resolve command
    resolve_parser = subparsers.add_parser("resolve", help="Run entity resolution")
    resolve_parser.add_argument("--documents", required=True, help="Processed documents directory")
    resolve_parser.add_argument("--transactions", required=True, help="Transactions file")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run full agent pipeline")
    run_parser.add_argument("--query", required=True, help="Query to process")
    run_parser.add_argument("--case-id", help="Case ID")
    run_parser.add_argument("--transaction-date", help="Transaction date (ISO format)")
    run_parser.add_argument("--votes", type=int, default=1, help="Number of self-consistency votes")
    run_parser.add_argument("--output", help="Output path for submission.json")

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument("--test-cases", required=True, help="Test cases file")
    eval_parser.add_argument("--golden", required=True, help="Golden dataset file")
    eval_parser.add_argument("--output-dir", help="Output directory for results")

    args = parser.parse_args()
    setup_logging(args.debug)

    # Dispatch
    commands = {
        "ingest": cmd_ingest,
        "retrieve": cmd_retrieve,
        "resolve": cmd_resolve,
        "run": cmd_run,
        "eval": cmd_eval,
    }

    if args.command in commands:
        return commands[args.command](args)
    else:
        logger.error(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())