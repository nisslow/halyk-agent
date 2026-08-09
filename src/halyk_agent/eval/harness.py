"""
Evaluation Harness for Halyk AI Challenge.
Validates agent output against golden dataset (ground_truth.json) and calculates exact score.
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from loguru import logger

@dataclass
class EvalMetrics:
    """Evaluation metrics per cell and total."""
    total_cells: int = 0
    evaluated_cells: int = 0
    total_score: float = 0.0
    max_possible_score: float = 0.0
    
    # Detailed breakdown per scenario -> covenant
    details: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    
    def print_report(self):
        logger.info("========== Halyk AI Challenge Evaluation Report ==========")
        logger.info(f"Total cells expected: {self.total_cells}")
        logger.info(f"Total cells evaluated: {self.evaluated_cells}")
        
        # Max score is 1.0 per cell
        logger.info(f"Final Score: {self.total_score:.4f} / {self.max_possible_score:.4f}")
        logger.info(f"Percentage: {(self.total_score / self.max_possible_score * 100) if self.max_possible_score > 0 else 0:.2f}%")
        logger.info("==========================================================")
        

class EvalHarness:
    """Main evaluation harness for the Hackathon rules."""

    def __init__(self, ground_truth_path: Path):
        self.ground_truth_path = ground_truth_path
        self.ground_truth_data = self._load_ground_truth()

    def _load_ground_truth(self) -> dict[str, Any]:
        if self.ground_truth_path.exists():
            with open(self.ground_truth_path, "r", encoding="utf-8") as f:
                return json.load(f)
        raise FileNotFoundError(f"Ground truth not found at {self.ground_truth_path}")

    def run_evaluation(self, submission_path: Path) -> EvalMetrics:
        """Run full evaluation on a submission.json file."""
        logger.info(f"Starting evaluation of {submission_path}...")

        if not submission_path.exists():
            logger.error("Submission file does not exist.")
            return EvalMetrics()

        with open(submission_path, "r", encoding="utf-8") as f:
            try:
                submission_data = json.load(f)
            except json.JSONDecodeError:
                logger.error("Submission is not a valid JSON. Score = 0")
                return EvalMetrics()

        answers = submission_data.get("answers", {})
        expected_scenarios = self.ground_truth_data.get("scenarios", {})

        metrics = EvalMetrics()
        
        for scenario_id, scenario_data in expected_scenarios.items():
            metrics.details[scenario_id] = {}
            covenants = scenario_data.get("covenants", {})
            
            for cov_id, gt_cell in covenants.items():
                metrics.total_cells += 1
                metrics.max_possible_score += 1.0
                
                # Retrieve submitted cell
                sub_cell = answers.get(scenario_id, {}).get(cov_id)
                
                cell_score, cell_details = self._evaluate_cell(gt_cell, sub_cell)
                metrics.total_score += cell_score
                metrics.evaluated_cells += 1
                metrics.details[scenario_id][cov_id] = cell_details
                
                if cell_score < 1.0:
                    logger.warning(f"Penalty in {scenario_id} [{cov_id}]: Score = {cell_score:.4f}. Details: {cell_details['reason']}")
                    
        return metrics

    def _evaluate_cell(self, gt: dict, sub: Optional[dict]) -> tuple[float, dict]:
        """Evaluates a single cell according to CASE.ru.md rules."""
        if not sub:
            return 0.0, {"score": 0.0, "reason": "Cell missing"}

        gt_status = gt.get("status")
        sub_status = sub.get("status")
        
        if sub_status not in ["COMPLIANT", "BREACH"]:
            return 0.0, {"score": 0.0, "reason": "Status not strictly COMPLIANT or BREACH"}
            
        if sub_status != gt_status:
            return 0.0, {"score": 0.0, "reason": f"Status mismatch: expected {gt_status}, got {sub_status}"}

        # Base status score
        score = 0.50
        reason_parts = ["Status matched (+0.50)"]

        # Evaluate actual (0.30 points)
        gt_actual = gt.get("actual")
        sub_actual = sub.get("actual")
        
        actual_decay_factor = 0.0
        
        if sub_actual is None or not isinstance(sub_actual, (int, float)):
            reason_parts.append("Actual missing or not a number (+0.0)")
        else:
            try:
                sub_actual = float(sub_actual)
                if sub_actual < 0:
                    reason_parts.append("Actual must be positive (+0.0)")
                elif gt_actual == 0:
                    if sub_actual == 0:
                        actual_decay_factor = 1.0
                    else:
                        actual_decay_factor = 0.0
                else:
                    error = abs(sub_actual - gt_actual) / abs(gt_actual)
                    actual_decay_factor = max(0.0, 1.0 - (error / 0.05))
            except (ValueError, TypeError):
                reason_parts.append("Actual is invalid (+0.0)")
                
        actual_score = 0.30 * actual_decay_factor
        score += actual_score
        reason_parts.append(f"Actual score (+{actual_score:.4f})")

        # Evaluate evidence_txn_id (0.20 points)
        gt_evidence = gt.get("evidence_txn_id")
        sub_evidence = sub.get("evidence_txn_id")
        
        evidence_score = 0.0
        if gt_evidence is None:
            # "Если в ключе стоит null, эти баллы убывают вместе с actual по той же шкале"
            evidence_score = 0.20 * actual_decay_factor
            reason_parts.append(f"Evidence null logic (+{evidence_score:.4f})")
        else:
            if sub_evidence == gt_evidence:
                evidence_score = 0.20
                reason_parts.append("Evidence matched (+0.20)")
            else:
                reason_parts.append(f"Evidence mismatched: expected {gt_evidence}, got {sub_evidence} (+0.0)")
                
        score += evidence_score
        
        return score, {
            "score": score,
            "reason": " | ".join(reason_parts),
            "expected": gt,
            "submitted": sub
        }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python harness.py <ground_truth.json> <submission.json>")
        sys.exit(1)
        
    gt_path = Path(sys.argv[1])
    sub_path = Path(sys.argv[2])
    
    harness = EvalHarness(gt_path)
    metrics = harness.run_evaluation(sub_path)
    metrics.print_report()