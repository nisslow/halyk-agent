"""
Proof Bundle Generator with Self-Consistency Voting and Counterfactual Analysis.
"""
from __future__ import annotations
import logging
import json
from collections import Counter
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from loguru import logger

from halyk_agent.config import settings
from halyk_agent.models import (
    ProofBundle,
    ReasoningStep,
    Evidence,
    CounterfactualResult,
    SubmissionRecord,
    SubmissionOutput,
    BoundingBox,
)
from halyk_agent.agents import HalykAgent, AgentState


class SelfConsistencyVoter:
    """Runs multiple reasoning traces and votes on the consensus."""

    def __init__(self, agent: HalykAgent, n_votes: int = 5):
        self.agent = agent
        self.n_votes = n_votes

    def vote(
        self,
        query: str,
        case_id: Optional[str] = None,
        transaction_date: Optional[datetime] = None,
    ) -> ProofBundle:
        """Run N traces and aggregate."""
        logger.info(f"Running self-consistency with N={self.n_votes}")

        traces = []
        decisions = []
        confidences = []
        proof_bundles = []

        for i in range(self.n_votes):
            logger.debug(f"Vote {i+1}/{self.n_votes}")
            try:
                bundle = self.agent.run(query, case_id, transaction_date)
                proof_bundles.append(bundle)
                decisions.append(bundle.decision)
                confidences.append(bundle.confidence)
                traces.append(bundle.reasoning_trace)
            except Exception as e:
                logger.error(f"Vote {i+1} failed: {e}")

        if not decisions:
            raise RuntimeError("All votes failed")

        # Majority vote on decision
        decision_counts = Counter(decisions)
        final_decision = decision_counts.most_common(1)[0][0]
        decision_agreement = decision_counts[final_decision] / len(decisions)

        # Average confidence for winning decision
        winning_confidences = [
            c for d, c in zip(decisions, confidences) if d == final_decision
        ]
        final_confidence = sum(winning_confidences) / len(winning_confidences) if winning_confidences else 0

        # Merge reasoning traces (take most common steps)
        merged_trace = self._merge_traces(traces)

        # Merge evidence
        merged_evidence = self._merge_evidence(proof_bundles)

        # Merge counterfactuals
        merged_counterfactuals = self._merge_counterfactuals(proof_bundles)

        return ProofBundle(
            query=query,
            decision=final_decision,
            confidence=final_confidence * decision_agreement,  # penalize disagreement
            reasoning_trace=merged_trace,
            evidence_bundle=merged_evidence,
            counterfactual_analysis=merged_counterfactuals,
            business_rule_validation={},  # from first bundle
            metadata={
                "self_consistency": {
                    "n_votes": self.n_votes,
                    "decision_agreement": decision_agreement,
                    "decision_distribution": dict(decision_counts),
                    "individual_confidences": confidences,
                },
                "case_id": case_id or str(uuid4()),
            },
        )

    def _merge_traces(self, traces: list[list[ReasoningStep]]) -> list[ReasoningStep]:
        """Merge multiple reasoning traces."""
        if not traces:
            return []

        # Align traces by step number
        max_steps = max(len(t) for t in traces)
        merged = []

        for step_num in range(1, max_steps + 1):
            step_claims = []
            step_evidence = []
            step_confidences = []
            step_methods = []

            for trace in traces:
                for step in trace:
                    if step.step_num == step_num:
                        step_claims.append(step.claim)
                        step_evidence.extend(step.evidence)
                        step_confidences.append(step.confidence)
                        step_methods.append(step.method)
                        break

            if step_claims:
                # Most common claim
                claim_counts = Counter(step_claims)
                final_claim = claim_counts.most_common(1)[0][0]
                claim_agreement = claim_counts[final_claim] / len(step_claims)

                merged.append(ReasoningStep(
                    step_num=step_num,
                    claim=final_claim,
                    reasoning=f"Consensus from {len(step_claims)} traces (agreement: {claim_agreement:.2f})",
                    evidence=step_evidence,
                    confidence=sum(step_confidences) / len(step_confidences) * claim_agreement,
                    method=step_methods[0] if step_methods else "consensus",
                ))

        return merged

    def _merge_evidence(self, bundles: list[ProofBundle]) -> dict[str, Any]:
        """Merge evidence from multiple bundles."""
        all_docs = set()
        all_calculations = {}
        all_verification = {}

        for bundle in bundles:
            all_docs.update(bundle.evidence_bundle.get("supporting_docs", []))
            all_calculations.update(bundle.evidence_bundle.get("calculations", {}))
            all_verification.update(bundle.evidence_bundle.get("verification", {}))

        return {
            "supporting_docs": list(all_docs),
            "calculations": all_calculations,
            "verification": all_verification,
        }

    def _merge_counterfactuals(self, bundles: list[ProofBundle]) -> list[CounterfactualResult]:
        """Merge counterfactual results."""
        # Group by removed element
        grouped = {}
        for bundle in bundles:
            for cf in bundle.counterfactual_analysis:
                key = cf.removed_element
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(cf)

        # Average results
        merged = []
        for key, cfs in grouped.items():
            merged.append(CounterfactualResult(
                removed_element=key,
                original_decision=cfs[0].original_decision,
                counterfactual_decision=Counter(
                    cf.counterfactual_decision for cf in cfs
                ).most_common(1)[0][0],
                original_confidence=sum(cf.original_confidence for cf in cfs) / len(cfs),
                counterfactual_confidence=sum(cf.counterfactual_confidence for cf in cfs) / len(cfs),
                decision_flipped=any(cf.decision_flipped for cf in cfs),
                confidence_delta=sum(cf.confidence_delta for cf in cfs) / len(cfs),
            ))

        return merged


class CounterfactualAnalyzer:
    """Analyzes robustness by removing evidence."""

    def __init__(self, agent: HalykAgent):
        self.agent = agent

    def analyze(
        self,
        query: str,
        baseline_bundle: ProofBundle,
        elements_to_test: list[str],
        case_id: Optional[str] = None,
        transaction_date: Optional[datetime] = None,
    ) -> list[CounterfactualResult]:
        """Run counterfactual analysis by removing elements."""
        logger.info(f"Running counterfactual analysis on {len(elements_to_test)} elements")

        results = []

        for element_id in elements_to_test:
            # Create modified query that excludes this element
            # In production, this would re-run the pipeline with the element removed
            # For now, simulate

            original_decision = baseline_bundle.decision
            original_confidence = baseline_bundle.confidence

            # Simulate counterfactual (placeholder)
            # Real implementation: mask element in retriever, re-run
            counterfactual_confidence = original_confidence * 0.85  # simulate degradation
            decision_flipped = counterfactual_confidence < 0.5

            if decision_flipped:
                counterfactual_decision = "REJECT" if original_decision == "APPROVE" else "APPROVE"
            else:
                counterfactual_decision = original_decision

            results.append(CounterfactualResult(
                removed_element=element_id,
                original_decision=original_decision,
                counterfactual_decision=counterfactual_decision,
                original_confidence=original_confidence,
                counterfactual_confidence=counterfactual_confidence,
                decision_flipped=decision_flipped,
                confidence_delta=original_confidence - counterfactual_confidence,
            ))

        return results


class ProofBundleGenerator:
    """Generates final proof bundles with full citations."""

    def __init__(self):
        pass

    def generate(
        self,
        query: str,
        decision: str,
        confidence: float,
        reasoning_trace: list[ReasoningStep],
        evidence: list[Evidence],
        counterfactuals: list[CounterfactualResult],
        business_rules: dict[str, Any],
        metadata: dict[str, Any],
    ) -> ProofBundle:
        """Generate a complete proof bundle."""
        return ProofBundle(
            query=query,
            decision=decision,
            confidence=confidence,
            reasoning_trace=reasoning_trace,
            evidence_bundle=self._format_evidence(evidence),
            counterfactual_analysis=counterfactuals,
            business_rule_validation=business_rules,
            metadata=metadata,
        )

    def _format_evidence(self, evidence: list[Evidence]) -> dict[str, Any]:
        """Format evidence for output."""
        docs = set()
        tables = set()
        transactions = set()
        calculations = []

        for e in evidence:
            if e.source_type == "text":
                docs.add(e.source_doc_id)
            elif e.source_type == "table":
                tables.add(e.table_id or e.source_doc_id)
            elif e.source_type == "transaction":
                transactions.add(e.txn_id)
            elif e.source_type == "calculation":
                calculations.append({
                    "formula": e.formula,
                    "inputs": e.inputs,
                    "confidence": e.confidence,
                })

        return {
            "supporting_docs": list(docs),
            "supporting_tables": list(tables),
            "supporting_transactions": list(transactions),
            "calculations": calculations,
        }

    def to_submission_json(self, bundles: list[ProofBundle], output_path: str) -> None:
        """Export proof bundles to submission.json format."""
        submissions = []

        for bundle in bundles:
            # Convert evidence to submission format
            evidence_list = []
            for step in bundle.reasoning_trace:
                for ev in step.evidence:
                    evidence_list.append({
                        "source": ev.source_doc_id,
                        "type": ev.source_type,
                        "page": ev.page,
                        "bbox": ev.bbox.to_dict() if ev.bbox else None,
                        "claim": step.claim,
                        "confidence": ev.confidence,
                    })

            record = SubmissionRecord(
                case_id=bundle.metadata.get("case_id", str(uuid4())),
                decision=bundle.decision,
                confidence=bundle.confidence,
                reasoning=self._trace_to_text(bundle.reasoning_trace),
                evidence=evidence_list,
            )
            submissions.append(record)

        output = SubmissionOutput(
            submissions=submissions,
            metadata={
                "generated_at": datetime.utcnow().isoformat(),
                "total_cases": len(submissions),
            },
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output.model_dump(), f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"Submission saved to {output_path}")

    def _trace_to_text(self, trace: list[ReasoningStep]) -> str:
        """Convert reasoning trace to human-readable text."""
        lines = []
        for step in trace:
            lines.append(f"Step {step.step_num}: {step.claim}")
            lines.append(f"  Reasoning: {step.reasoning}")
            lines.append(f"  Confidence: {step.confidence:.2f}")
            if step.evidence:
                lines.append(f"  Evidence: {len(step.evidence)} citations")
        return "\n".join(lines)


def create_voter(agent: HalykAgent, n_votes: int = 5) -> SelfConsistencyVoter:
    """Factory for self-consistency voter."""
    return SelfConsistencyVoter(agent, n_votes)


def create_counterfactual_analyzer(agent: HalykAgent) -> CounterfactualAnalyzer:
    """Factory for counterfactual analyzer."""
    return CounterfactualAnalyzer(agent)


def create_proof_generator() -> ProofBundleGenerator:
    """Factory for proof bundle generator."""
    return ProofBundleGenerator()