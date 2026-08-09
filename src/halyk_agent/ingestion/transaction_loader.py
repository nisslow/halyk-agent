"""
Transaction Registry Loader.
Loads and normalizes transaction data from CSV/Excel/Parquet.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import polars as pl
from loguru import logger

from halyk_agent.config import settings
from halyk_agent.models import Transaction

# Column mapping for different possible formats
COLUMN_MAPPING = {
    "txn_id": ["txn_id", "transaction_id", "id", "transactionId"],
    "date": ["date", "transaction_date", "datetime", "txn_date", "date_time"],
    "amount": ["amount", "sum", "value", "transaction_amount", "amt"],
    "currency": ["currency", "ccy", "cur", "currency_code"],
    "sender": ["sender", "from", "sender_name", "payer", "originator"],
    "receiver": ["receiver", "to", "receiver_name", "payee", "beneficiary"],
    "sender_bin": ["sender_bin", "sender_inn", "payer_bin", "originator_bin", "bin_from"],
    "receiver_bin": ["receiver_bin", "receiver_inn", "payee_bin", "beneficiary_bin", "bin_to"],
    "purpose": ["purpose", "description", "memo", "payment_purpose", "details"],
    "category": ["category", "type", "transaction_type", "txn_category"],
}


class TransactionLoader:
    """Loads and normalizes transaction registry."""

    def __init__(self):
        self.required_columns = ["txn_id", "date", "amount", "sender", "receiver"]

    def load(self, path: Path) -> list[Transaction]:
        """Load transactions from file."""
        logger.info(f"Loading transactions from: {path}")

        # Read based on extension
        if path.suffix.lower() in [".parquet", ".pq"]:
            df = pl.read_parquet(path)
        elif path.suffix.lower() in [".csv"]:
            df = pl.read_csv(path, try_parse_dates=True)
        elif path.suffix.lower() in [".xlsx", ".xls"]:
            df = pl.read_excel(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        # Normalize columns
        df = self._normalize_columns(df)

        # Validate required columns
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Convert to Transaction objects
        transactions = []
        for row in df.iter_rows(named=True):
            try:
                txn = self._row_to_transaction(row)
                transactions.append(txn)
            except Exception as e:
                logger.warning(f"Failed to parse transaction row: {e}")

        logger.info(f"Loaded {len(transactions)} transactions")
        return transactions

    def _normalize_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Normalize column names to standard format."""
        col_map = {}
        df_columns_lower = {c.lower(): c for c in df.columns}

        for std_col, variants in COLUMN_MAPPING.items():
            for variant in variants:
                if variant.lower() in df_columns_lower:
                    col_map[df_columns_lower[variant.lower()]] = std_col
                    break

        return df.rename(col_map)

    def _row_to_transaction(self, row: dict[str, Any]) -> Transaction:
        """Convert row dict to Transaction."""
        # Parse date
        date_val = row["date"]
        if isinstance(date_val, str):
            date_val = pd.to_datetime(date_val).to_pydatetime()
        elif hasattr(date_val, "to_pydatetime"):
            date_val = date_val.to_pydatetime()

        # Parse amount
        amount = float(row["amount"])

        return Transaction(
            txn_id=str(row["txn_id"]),
            date=date_val,
            amount=amount,
            currency=str(row.get("currency", "KZT")),
            sender=str(row["sender"]),
            receiver=str(row["receiver"]),
            sender_bin=str(row["sender_bin"]) if row.get("sender_bin") else None,
            receiver_bin=str(row["receiver_bin"]) if row.get("receiver_bin") else None,
            purpose=str(row["purpose"]) if row.get("purpose") else None,
            category=str(row["category"]) if row.get("category") else None,
            raw_data=row,
        )

    def load_to_dataframe(self, path: Path) -> pl.DataFrame:
        """Load as Polars DataFrame for SQL queries."""
        transactions = self.load(path)
        return pl.DataFrame([txn.model_dump() for txn in transactions])

    def save_parquet(self, transactions: list[Transaction], path: Path):
        """Save transactions to Parquet."""
        df = pl.DataFrame([txn.model_dump() for txn in transactions])
        df.write_parquet(path)
        logger.info(f"Saved {len(transactions)} transactions to {path}")


def load_transactions(path: Path) -> list[Transaction]:
    """Convenience function to load transactions."""
    loader = TransactionLoader()
    return loader.load(path)