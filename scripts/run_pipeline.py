#!/usr/bin/env python3
"""
End-to-End Pipeline Orchestration for Halyk Agent.
"""
from __future__ import annotations
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger

from halyk_agent.ingestion import PDFIngestionPipeline, TransactionLoader, process_directory
from halyk_agent.graph import create_graph, create_resolver
from halyk_agent.agents import create_agent
from halyk_agent.proof import create_voter, create_proof_generator
from halyk_agent.eval import create_harness


def run_ingestion(input_dir: Path, output_dir: Path, transactions_path: Path = None):
    """Step 1: Ingest documents and transactions."""
    logger.info("=" * 60)
    logger.info("STEP 1: INGESTION")
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process PDFs
    logger.info(f"Processing PDFs from {input_dir}")
    parsed_docs = process_directory(input_dir, output_dir)
    logger.info(f"Processed {len(parsed_docs)} documents")

    # Process transactions
    if transactions_path and transactions_path.exists():
        logger.info(f"Loading transactions from {transactions_path}")
        loader = TransactionLoader()
        transactions = loader.load(transactions_path)
        loader.save_parquet(transactions, output_dir / "transactions.parquet")
        logger.info(f"Saved {len(transactions)} transactions")

    return parsed_docs


def run_entity_resolution(processed_dir: Path):
    """Step 2: Entity resolution."""
    logger.info("=" * 60)
    logger.info("STEP 2: ENTITY RESOLUTION")
    logger.info("=" * 60)

    # Load parsed documents
    # This would load from processed_dir and run resolution
    logger.info("Entity resolution requires loaded parsed docs - run via CLI")
    return []


def run_agent_query(
    query: str,
    case_id: str,
    transaction_date: str = None,
    votes: int = 5,
    output_path: Path = None,
):
    """Step 3: Run agent with self-consistency voting."""
    logger.info("=" * 60)
    logger.info("STEP 3: AGENT PIPELINE")
    logger.info("=" * 60)

    agent = create_agent()
    voter = create_voter(agent, n_votes=votes)

    txn_date = None
    if transaction_date:
        txn_date = datetime.fromisoformat(transaction_date)

    logger.info(f"Running agent for case {case_id} with {votes} votes...")
    bundle = voter.vote(
        query=query,
        case_id=case_id,
        transaction_date=txn_date,
    )

    # Output results
    print(f"\n{'='*60}")
    print(f"DECISION: {bundle.decision}")
    print(f"CONFIDENCE: {bundle.confidence:.4f}")
    print(f"AGREEMENT: {bundle.metadata.get('self_consistency', {}).get('decision_agreement', 0):.2f}")
    print(f"{'='*60}")

    # Save submission
    if output_path:
        generator = create_proof_generator()
        generator.to_submission_json([bundle], str(output_path))
        logger.info(f"Submission saved to {output_path}")

    return bundle


def run_evaluation(test_cases: Path, golden: Path, output_dir: Path):
    """Step 4: Run evaluation."""
    logger.info("=" * 60)
    logger.info("STEP 4: EVALUATION")
    logger.info("=" * 60)

    agent = create_agent()
    harness = create_harness(agent, golden, test_cases)

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

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Halyk Agent Pipeline")
    parser.add_argument("--step", choices=["ingest", "resolve", "run", "eval", "all"], default="all")
    parser.add_argument("--input-dir", default="data/raw", help="Raw PDF directory")
    parser.add_argument("--output-dir", default="data/processed", help="Processed output directory")
    parser.add_argument("--transactions", help="Transaction registry file")
    parser.add_argument("--query", help="Query for agent run")
    parser.add_argument("--case-id", help="Case ID")
    parser.add_argument("--transaction-date", help="Transaction date (ISO)")
    parser.add_argument("--votes", type=int, default=5, help="Self-consistency votes")
    parser.add_argument("--test-cases", default="data/test_cases.json", help="Test cases file")
    parser.add_argument("--golden", default="data/golden.json", help="Golden dataset")
    parser.add_argument("--eval-output", default="results/", help="Evaluation output dir")
    parser.add_argument("--output", help="Output submission.json path")

    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    base_path = Path(__file__).parent.parent

    if args.step in ["ingest", "all"]:
        run_ingestion(
            base_path / args.input_dir,
            base_path / args.output_dir,
            base_path / args.transactions if args.transactions else None,
        )

    if args.step in ["resolve", "all"]:
        run_entity_resolution(base_path / args.output_dir)

    if args.step in ["run", "all"]:
        if not args.query:
            logger.error("--query required for run step")
            return 1
        run_agent_query(
            query=args.query,
            case_id=args.case_id or "CASE_AUTO",
            transaction_date=args.transaction_date,
            votes=args.votes,
            output_path=base_path / args.output if args.output else None,
        )

    if args.step in ["eval", "all"]:
        run_evaluation(
            base_path / args.test_cases,
            base_path / args.golden,
            base_path / args.eval_output,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())