"""
Z3 Constraint Validator for Business Rules.
Proves compliance with financial regulations using SMT solving.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any, Optional

from z3 import (
    Solver, Real, Int, Bool, And, Or, Not, Implies,
    sat, unsat, unknown, Sum, If
)
from loguru import logger

from halyk_agent.config import settings


@dataclass
class VerificationResult:
    """Result of Z3 verification."""
    valid: bool
    rule_name: str
    error: Optional[str] = None
    model: Optional[dict[str, Any]] = None
    proof: Optional[str] = None


class Z3Validator:
    """Validates business rules using Z3 SMT solver."""

    def __init__(self):
        self.timeout_ms = settings.validation.z3_timeout_ms
        self.rules = self._load_rules()

    def _load_rules(self) -> dict[str, callable]:
        """Load business rules as Z3 constraints."""
        return {
            "max_commission_rate": self._rule_max_commission_rate,
            "min_reserve_ratio": self._rule_min_reserve_ratio,
            "max_loan_to_value": self._rule_max_loan_to_value,
            "kyc_required": self._rule_kyc_required,
            "aml_threshold": self._rule_aml_threshold,
            "interest_rate_cap": self._rule_interest_rate_cap,
            "transaction_limit": self._rule_transaction_limit,
            "currency_control": self._rule_currency_control,
        }

    def verify_calculation(self, name: str, calc: dict[str, Any]) -> dict[str, Any]:
        """Verify a calculation against relevant rules."""
        results = {}

        # Determine which rules apply based on calculation type
        applicable_rules = self._get_applicable_rules(name, calc)

        for rule_name in applicable_rules:
            if rule_name in self.rules:
                result = self.rules[rule_name](calc)
                results[rule_name] = {
                    "valid": result.valid,
                    "error": result.error,
                    "model": result.model,
                }

        return {
            "rules_checked": list(results.keys()),
            "all_valid": all(r["valid"] for r in results.values()),
            "details": results,
        }

    def _get_applicable_rules(self, name: str, calc: dict) -> list[str]:
        """Determine which rules apply to a calculation."""
        # Simple heuristic based on calculation name/content
        applicable = []

        if "commission" in name.lower() or "fee" in name.lower():
            applicable.append("max_commission_rate")
        if "loan" in name.lower() or "credit" in name.lower():
            applicable.extend(["max_loan_to_value", "interest_rate_cap"])
        if "reserve" in name.lower() or "capital" in name.lower():
            applicable.append("min_reserve_ratio")
        if "transaction" in name.lower() or "transfer" in name.lower():
            applicable.extend(["aml_threshold", "transaction_limit", "currency_control"])
        if "kyc" in name.lower() or "client" in name.lower():
            applicable.append("kyc_required")

        return applicable

    # ==================== BUSINESS RULES ====================

    def _rule_max_commission_rate(self, calc: dict) -> VerificationResult:
        """Rule: Commission rate cannot exceed allowed_rate from tariff."""
        s = Solver()
        s.set("timeout", self.timeout_ms)

        # Variables
        commission = Real("commission")
        amount = Real("amount")
        rate = Real("rate")

        # Constraints from calculation
        s.add(commission == calc.get("value", 0))
        s.add(amount == calc.get("inputs", {}).get("amount", 1))
        s.add(rate == commission / amount)

        # Business rule: rate <= allowed_rate (dynamic from document)
        allowed_rate_val = calc.get("inputs", {}).get("allowed_rate", 0.02)
        s.add(rate <= allowed_rate_val)

        return self._check_solver(s, "max_commission_rate", {"commission": commission, "amount": amount, "rate": rate})

    def _rule_min_reserve_ratio(self, calc: dict) -> VerificationResult:
        """Rule: Reserve ratio must be at least 10%."""
        s = Solver()
        s.set("timeout", self.timeout_ms)

        reserves = Real("reserves")
        liabilities = Real("liabilities")
        ratio = Real("ratio")

        s.add(reserves == calc.get("inputs", {}).get("reserves", 1))
        s.add(liabilities == calc.get("inputs", {}).get("liabilities", 1))
        s.add(ratio == reserves / liabilities)

        # Business rule: ratio >= 0.10 (10%)
        s.add(ratio >= 0.10)

        return self._check_solver(s, "min_reserve_ratio", {"reserves": reserves, "liabilities": liabilities, "ratio": ratio})

    def _rule_max_loan_to_value(self, calc: dict) -> VerificationResult:
        """Rule: Loan-to-value ratio cannot exceed 80%."""
        s = Solver()
        s.set("timeout", self.timeout_ms)

        loan_amount = Real("loan_amount")
        collateral_value = Real("collateral_value")
        ltv = Real("ltv")

        s.add(loan_amount == calc.get("inputs", {}).get("loan_amount", 1))
        s.add(collateral_value == calc.get("inputs", {}).get("collateral_value", 1))
        s.add(ltv == loan_amount / collateral_value)

        # Business rule: LTV <= 0.80 (80%)
        s.add(ltv <= 0.80)

        return self._check_solver(s, "max_loan_to_value", {"loan_amount": loan_amount, "collateral_value": collateral_value, "ltv": ltv})

    def _rule_interest_rate_cap(self, calc: dict) -> VerificationResult:
        """Rule: Interest rate cannot exceed central bank rate + 5%."""
        s = Solver()
        s.set("timeout", self.timeout_ms)

        interest_rate = Real("interest_rate")
        base_rate = Real("base_rate")

        s.add(interest_rate == calc.get("value", 0))
        s.add(base_rate == calc.get("inputs", {}).get("base_rate", 0.15))  # 15% default

        # Business rule: interest_rate <= base_rate + 0.05
        s.add(interest_rate <= base_rate + 0.05)

        return self._check_solver(s, "interest_rate_cap", {"interest_rate": interest_rate, "base_rate": base_rate})

    def _rule_kyc_required(self, calc: dict) -> VerificationResult:
        """Rule: KYC required for transactions > 500,000 KZT."""
        s = Solver()
        s.set("timeout", self.timeout_ms)

        amount = Real("amount")
        kyc_done = Bool("kyc_done")

        s.add(amount == calc.get("inputs", {}).get("amount", 0))
        # KYC status from calculation context
        kyc_status = calc.get("inputs", {}).get("kyc_completed", False)
        s.add(kyc_done == kyc_status)

        # Business rule: if amount > 500000 then kyc_done == True
        s.add(Implies(amount > 500000, kyc_done))

        return self._check_solver(s, "kyc_required", {"amount": amount, "kyc_done": kyc_done})

    def _rule_aml_threshold(self, calc: dict) -> VerificationResult:
        """Rule: AML reporting required for transactions > 2,000,000 KZT."""
        s = Solver()
        s.set("timeout", self.timeout_ms)

        amount = Real("amount")
        aml_reported = Bool("aml_reported")

        s.add(amount == calc.get("inputs", {}).get("amount", 0))
        aml_status = calc.get("inputs", {}).get("aml_reported", False)
        s.add(aml_reported == aml_status)

        # Business rule: if amount > 2000000 then aml_reported == True
        s.add(Implies(amount > 2000000, aml_reported))

        return self._check_solver(s, "aml_threshold", {"amount": amount, "aml_reported": aml_reported})

    def _rule_transaction_limit(self, calc: dict) -> VerificationResult:
        """Rule: Single transaction limit for individuals: 1,000,000 KZT/day."""
        s = Solver()
        s.set("timeout", self.timeout_ms)

        daily_total = Real("daily_total")
        is_individual = Bool("is_individual")

        s.add(daily_total == calc.get("inputs", {}).get("daily_total", 0))
        s.add(is_individual == calc.get("inputs", {}).get("is_individual", True))

        # Business rule: if is_individual then daily_total <= 1000000
        s.add(Implies(is_individual, daily_total <= 1000000))

        return self._check_solver(s, "transaction_limit", {"daily_total": daily_total, "is_individual": is_individual})

    def _rule_currency_control(self, calc: dict) -> VerificationResult:
        """Rule: FX transactions require special license for amounts > 10,000 USD equivalent."""
        s = Solver()
        s.set("timeout", self.timeout_ms)

        fx_amount_usd = Real("fx_amount_usd")
        has_license = Bool("has_license")

        s.add(fx_amount_usd == calc.get("inputs", {}).get("fx_amount_usd", 0))
        s.add(has_license == calc.get("inputs", {}).get("has_fx_license", False))

        # Business rule: if fx_amount_usd > 10000 then has_license == True
        s.add(Implies(fx_amount_usd > 10000, has_license))

        return self._check_solver(s, "currency_control", {"fx_amount_usd": fx_amount_usd, "has_license": has_license})

    def _check_solver(
        self,
        solver: Solver,
        rule_name: str,
        variables: dict[str, Any]
    ) -> VerificationResult:
        """Run Z3 solver and return result."""
        result = solver.check()

        if result == sat:
            model = solver.model()
            model_dict = {}
            for var in variables.values():
                if model.eval(var) is not None:
                    model_dict[str(var)] = str(model.eval(var))
            return VerificationResult(
                valid=True,
                rule_name=rule_name,
                model=model_dict,
                proof=f"SAT: {model_dict}",
            )
        elif result == unsat:
            return VerificationResult(
                valid=False,
                rule_name=rule_name,
                error="Constraint violated (UNSAT)",
                proof="UNSAT: Business rule violated",
            )
        else:
            return VerificationResult(
                valid=False,
                rule_name=rule_name,
                error=f"Solver returned {result} (timeout or unknown)",
                proof=f"UNKNOWN: {result}",
            )

    def verify_all_rules(self, context: dict[str, Any]) -> dict[str, VerificationResult]:
        """Verify all rules against a context."""
        results = {}
        for name, rule_fn in self.rules.items():
            try:
                results[name] = rule_fn(context)
            except Exception as e:
                logger.error(f"Rule {name} failed: {e}")
                results[name] = VerificationResult(
                    valid=False,
                    rule_name=name,
                    error=str(e),
                )
        return results


class CalculationEngine:
    """Safe calculation engine using numexpr/simpleeval."""

    def __init__(self):
        import numexpr as ne
        self.ne = ne

        from simpleeval import SimpleEval
        self.simple_eval = SimpleEval()
        # Add safe functions
        self.simple_eval.functions.update({
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
        })

    def evaluate(self, expression: str, variables: dict[str, Any]) -> float:
        """Safely evaluate a mathematical expression."""
        # Try numexpr first (faster for array ops)
        try:
            result = self.ne.evaluate(expression, local_dict=variables)
            return float(result)
        except Exception:
            pass

        # Fallback to simpleeval
        try:
            return float(self.simple_eval.eval(expression, variables))
        except Exception as e:
            raise ValueError(f"Calculation failed: {e}")

    def evaluate_formula(self, formula: str, variables: dict[str, Any]) -> dict[str, Any]:
        """Evaluate formula and return detailed result."""
        try:
            value = self.evaluate(formula, variables)
            return {
                "value": value,
                "formula": formula,
                "variables": variables,
                "success": True,
            }
        except Exception as e:
            return {
                "value": None,
                "formula": formula,
                "variables": variables,
                "success": False,
                "error": str(e),
            }