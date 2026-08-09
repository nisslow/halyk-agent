import os
import sys
import json
import re
from pathlib import Path
import pandas as pd
from loguru import logger

def main():
    dataset_dir = Path(os.getenv("DATASET_DIR", "E:/AntigravityProjects/halyk-agent/agentic-bank-public"))
    ledger_path = dataset_dir / "master_ledger_2025.csv"
    raw_data_dir = dataset_dir / "documents"
    output_mapping_path = Path(__file__).parent.parent / "data" / "scenario_mapping.json"
    
    if not ledger_path.exists():
        logger.error("Ledger not found")
        return
        
    if not raw_data_dir.exists():
        logger.error("Raw data dir not found. Has ingestion finished?")
        return
        
    # 1. Build Account -> Scenario mapping from ledger
    df = pd.read_csv(ledger_path)
    account_to_scenario = {}
    for _, row in df.iterrows():
        acc = row['account_id']
        txn = row['txn_id']
        scenario = txn.split('-')[1]
        account_to_scenario[acc] = scenario
        
    logger.info(f"Built mapping for {len(account_to_scenario)} accounts.")
    
    # 2. Scan PDF files
    scenario_to_docs = {}
    
    pdf_files = list(raw_data_dir.glob("*.pdf"))
    logger.info(f"Scanning {len(pdf_files)} PDF files for Account IDs...")
    
    import fitz  # PyMuPDF
    
    for pdf_path in pdf_files:
        doc_id = pdf_path.stem
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        except Exception as e:
            logger.error(f"Failed to read {pdf_path}: {e}")
            continue
            
        # Find all ACC-XXXX mentions
        matches = set(re.findall(r"ACC-\d{4}", text))
        
        for acc in matches:
            if acc in account_to_scenario:
                scenario = account_to_scenario[acc]
                if scenario not in scenario_to_docs:
                    scenario_to_docs[scenario] = []
                
                # Check if we already have this doc_id for this scenario
                if doc_id not in scenario_to_docs[scenario]:
                    scenario_to_docs[scenario].append(doc_id)
                    logger.info(f"Mapped {doc_id}.pdf to Scenario {scenario} via {acc}")
                    
    # Save mapping
    with open(output_mapping_path, "w", encoding="utf-8") as f:
        json.dump(scenario_to_docs, f, indent=2)
        
    logger.info(f"Saved mapping to {output_mapping_path}")
    
    # Print summary
    for scenario, docs in scenario_to_docs.items():
        logger.info(f"Scenario {scenario}: {len(docs)} documents")

if __name__ == "__main__":
    main()
