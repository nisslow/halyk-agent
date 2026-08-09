#!/usr/bin/env python3
"""
Generate test data for Halyk Agent.
"""
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4


def generate_test_cases(output_path: Path, n_cases: int = 10):
    """Generate synthetic test cases."""
    decisions = ["APPROVE", "REJECT", "REVIEW"]
    queries = [
        "Should transaction {txn_id} be approved?",
        "Is the commission for {txn_id} compliant with regulations?",
        "Does {txn_id} violate AML thresholds?",
        "Is KYC required for transaction {txn_id}?",
        "What is the maximum loan-to-value for {txn_id}?",
    ]

    cases = []
    for i in range(n_cases):
        txn_id = f"TXN_{uuid4().hex[:8].upper()}"
        case = {
            "case_id": f"CASE_{i+1:03d}",
            "query": random.choice(queries).format(txn_id=txn_id),
            "transaction_date": (datetime.now() - timedelta(days=random.randint(0, 365))).isoformat(),
            "expected_decision": random.choice(decisions),
            "metadata": {
                "txn_id": txn_id,
                "amount": random.randint(10000, 5000000),
                "currency": "KZT",
            },
        }
        cases.append(case)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"cases": cases}, f, ensure_ascii=False, indent=2)

    print(f"Generated {n_cases} test cases to {output_path}")


def generate_golden(output_path: Path, n_cases: int = 10):
    """Generate golden dataset."""
    cases = []
    for i in range(n_cases):
        cases.append({
            "case_id": f"CASE_{i+1:03d}",
            "correct_decision": "APPROVE" if i % 2 == 0 else "REJECT",
            "key_regulations": ["Reg_2023_001", "Reg_2024_002"],
            "key_transactions": [f"TXN_{uuid4().hex[:8].upper()}"],
            "expected_reasoning": "Based on regulation X, the transaction...",
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"cases": cases}, f, ensure_ascii=False, indent=2)

    print(f"Generated golden dataset to {output_path}")


def generate_transactions(output_path: Path, n_txns: int = 100):
    """Generate synthetic transaction registry."""
    import pandas as pd

    categories = ["commission", "transfer", "loan_payment", "fx", "fee"]
    statuses = ["completed", "pending", "failed"]

    data = []
    for i in range(n_txns):
        data.append({
            "txn_id": f"TXN_{uuid4().hex[:8].upper()}",
            "date": datetime.now() - timedelta(days=random.randint(0, 365)),
            "amount": round(random.uniform(1000, 1000000), 2),
            "currency": "KZT",
            "sender": f"Org_{random.randint(1, 20)}",
            "receiver": f"Org_{random.randint(1, 20)}",
            "sender_bin": f"{random.randint(10**11, 10**12-1)}",
            "receiver_bin": f"{random.randint(10**11, 10**12-1)}",
            "purpose": f"Payment for {random.choice(categories)}",
            "category": random.choice(categories),
            "status": random.choice(statuses),
        })

    df = pd.DataFrame(data)
    df.to_parquet(output_path, index=False)
    print(f"Generated {n_txns} transactions to {output_path}")


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    generate_test_cases(data_dir / "test_cases.json", 20)
    generate_golden(data_dir / "golden.json", 20)
    generate_transactions(data_dir / "transactions.parquet", 200)