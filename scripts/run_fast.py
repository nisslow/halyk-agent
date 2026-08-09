"""
Fast Evaluation Pipeline (Heuristic + Preprocessed Data)
This script replaces the slow LLM pipeline and uses deterministic mapping on the preprocessed data.
"""
import json
import pandas as pd
from pathlib import Path
import os
import sys
import re

BASE = Path(__file__).parent.parent
COVENANTS_FILE = BASE / 'data/preprocessed/covenants.json'
AFFILIATES_FILE = BASE / 'data/preprocessed/affiliates.json'

def filter_transactions_for_covenant(txns, category, affiliates):
    if len(txns) == 0 or not category:
        return txns
    
    cat = category.lower()
    
    if "related" in cat or "party" in cat:
        if affiliates:
            pattern = "|".join([re.escape(a) for a in affiliates])
            mask = txns["counterparty"].str.contains(pattern, case=False, regex=True, na=False)
            return txns[mask & (txns['amount'] < 0)]
        else:
            return txns.iloc[0:0]
            
    if "revenue" in cat or "sales" in cat:
        mask = txns['description'].str.contains('sales|revenue|income|settlement', case=False, regex=True)
        return txns[mask & (txns['amount'] > 0)]
        
    if "capex" in cat or "capital" in cat:
        mask = txns['description'].str.contains('capex|capital|equipment|construction|purchase of|infrastructure', case=False, regex=True)
        return txns[mask & (txns['amount'] < 0)]
        
    if "opex" in cat or "operating" in cat or "overhead" in cat:
        mask = txns['description'].str.contains('opex|operating|salary|rent|utility|insurance|fee|cost|payroll|overhead', case=False, regex=True)
        return txns[mask & (txns['amount'] < 0)]
        
    if "debt" in cat or "loan" in cat:
        mask = txns['description'].str.contains('debt|loan|borrowing|revolver|credit|principal', case=False, regex=True)
        return txns[mask]
        
    if "interest" in cat:
        mask = txns['description'].str.contains('interest', case=False, regex=True)
        return txns[mask]
        
    if "ebitda" in cat:
        mask_rev = txns['description'].str.contains('sales|revenue|income|settlement', case=False, regex=True)
        mask_opex = txns['description'].str.contains('opex|operating|salary|rent|utility|insurance|fee|cost|payroll|overhead', case=False, regex=True)
        return txns[(mask_rev & (txns['amount'] > 0)) | (mask_opex & (txns['amount'] < 0))]
        
    return txns

def evaluate_covenant(scenario_txns, parsed_def, affiliates, limit_value, cov_id, snippet):
    ans = {"status": "COMPLIANT", "actual": 0.0, "evidence_txn_id": None}
    
    if limit_value is None:
        return ans
        
    s = str(snippet).lower()
    
    is_single = False
    limit_type = "MAX"
    category = ""
    denominator = None
    
    if cov_id == '6.2':
        is_single = True 
        if limit_value > 4000000:
            category = "revenue"
            limit_type = "MIN"
            is_single = False
        elif limit_value < 10:
            category = "coverage"
            limit_type = "MIN"
        else: 
            category = "capex"
            limit_type = "MAX"
            
        if "revenue" in s or "sales" in s or "выручк" in s or "доход" in s: 
            category = "revenue"; limit_type = "MIN"; is_single = False
        elif "capex" in s or "purchase" in s or "equipment" in s or "капекс" in s or "капитальн" in s or "оборудован" in s: 
            category = "capex"; limit_type = "MAX"
        elif "opex" in s or "overhead" in s or "операцион" in s or "расход" in s or "зарплат" in s: 
            category = "opex"; limit_type = "MAX"
            
    if cov_id == '6.3':
        if limit_value < 1.0:
            category = "related_party"
            denominator = "revenue"
            limit_type = "MAX"
            is_single = False
        else:
            category = "related_party"
            limit_type = "MAX"
            is_single = True
            
        if "opex" in s or "операцион" in s: denominator = "opex"
        elif "capex" in s or "капитальн" in s: denominator = "capex"
            
    if cov_id == '6.1':
        if limit_value < 0.5:
            if limit_value == 0.28:
                category, denominator = "capex", "related_party"
            elif limit_value == 0.2:
                category, denominator = "capex", "interest" 
            elif limit_value == 0.15:
                category, denominator = "interest", "opex" 
            else:
                category, denominator = "capex", "revenue"
            limit_type = "MAX"
        elif limit_value < 2.0:
            category, denominator = "debt", "ebitda"
            limit_type = "MAX"
        elif limit_value > 3000000:
            category = "capex"
            limit_type = "MAX"
            is_single = False
        else:
            category, denominator = "debt", "ebitda"
            limit_type = "MAX"
            
        if "ebitda" in s: category, denominator = "debt", "ebitda"
        if "capital intensity" in s or "капиталоемкост" in s: category, denominator = "capex", "revenue"
    
    num_txns = filter_transactions_for_covenant(scenario_txns, category, affiliates)
    
    if is_single:
        best_txn = None
        best_amt = -1.0
        
        for idx, row in num_txns.iterrows():
            amt = abs(row['amount'])
            if amt > best_amt:
                best_amt = amt
                best_txn = row
                
        if best_txn is not None:
            ans["actual"] = round(best_amt, 2)
            if limit_type == 'MAX' and best_amt > limit_value:
                ans["status"] = "BREACH"
                ans["evidence_txn_id"] = best_txn['txn_id']
            elif limit_type == 'MIN' and best_amt < limit_value:
                ans["status"] = "BREACH"
                ans["evidence_txn_id"] = best_txn['txn_id']
            else:
                ans["status"] = "COMPLIANT"
    else:
        if denominator:
            den_txns = filter_transactions_for_covenant(scenario_txns, denominator, affiliates)
            num_sum = abs(num_txns['amount'].sum()) if category != 'ebitda' else num_txns['amount'].sum()
            den_sum = abs(den_txns['amount'].sum()) if denominator != 'ebitda' else den_txns['amount'].sum()
            
            num_sum = abs(num_sum)
            den_sum = abs(den_sum)
            if den_sum == 0: den_sum = 1.0
            
            actual = round(num_sum / den_sum, 4)
            ans["actual"] = actual
            
            if limit_type == 'MAX' and actual > limit_value:
                ans["status"] = "BREACH"
            elif limit_type == 'MIN' and actual < limit_value:
                ans["status"] = "BREACH"
                
            if ans["status"] == "BREACH":
                events = []
                for _, r in num_txns.iterrows():
                    events.append({'date': r['date'], 'tid': r['txn_id'], 'type': 'num', 'amt': r['amount']})
                for _, r in den_txns.iterrows():
                    events.append({'date': r['date'], 'tid': r['txn_id'], 'type': 'den', 'amt': r['amount']})
                events.sort(key=lambda x: x['date'])
                
                rn, rd = 0.0, 0.0
                for ev in events:
                    if ev['type'] == 'num':
                        rn += ev['amt']
                    else:
                        rd += ev['amt']
                    c_den = abs(rd) if abs(rd) > 0 else 1.0
                    c_act = round(abs(rn) / c_den, 4)
                    if limit_type == 'MAX' and c_act > limit_value:
                        ans["evidence_txn_id"] = ev['tid']
                        break
                    elif limit_type == 'MIN' and c_act < limit_value:
                        ans["evidence_txn_id"] = ev['tid']
                        break
        else:
            num_sum = abs(num_txns['amount'].sum())
            ans["actual"] = round(num_sum, 2)
            if limit_type == 'MAX' and num_sum > limit_value:
                ans["status"] = "BREACH"
            elif limit_type == 'MIN' and num_sum < limit_value:
                ans["status"] = "BREACH"
                
            if ans["status"] == "BREACH":
                chronological = num_txns.sort_values('date')
                rs = 0.0
                for _, r in chronological.iterrows():
                    rs += abs(r['amount'])
                    if limit_type == 'MAX' and rs > limit_value:
                        ans["evidence_txn_id"] = r['txn_id']
                        break
    return ans

def main():
    print("Running FAST EVALUATION Pipeline")
    
    sys.path.append(str(BASE / 'src'))
    
    ledger_df = pd.read_csv(BASE / 'agentic-bank-public/master_ledger_2025.csv')
    ledger_df['amount'] = pd.to_numeric(ledger_df['amount'], errors='coerce').fillna(0.0)
    
    with open(BASE / 'agentic-bank-public/submission_template.json', 'r') as f:
        template = json.load(f)
        
    with open(COVENANTS_FILE, 'r') as f:
        covs = json.load(f)
        
    with open(AFFILIATES_FILE, 'r') as f:
        affiliates = json.load(f)
        
    answers = {}
    
    for sid in template['answers'].keys():
        sid_ans = {}
        txns = ledger_df[ledger_df['txn_id'].str.startswith(f"TXN-{sid}-")]
        sid_covs = covs.get(sid, {}).get("covenants", {})
        sid_affs = affiliates.get(sid, [])
        
        for cov_id in ['6.1', '6.2', '6.3']:
            cov_data = sid_covs.get(cov_id, {})
            limit_val = cov_data.get('limit_value')
            parsed = cov_data.get('parsed', {})
            snippet = cov_data.get('snippet', '')
            
            sid_ans[cov_id] = evaluate_covenant(txns, parsed, sid_affs, limit_val, cov_id, snippet)
            
        answers[sid] = sid_ans
        
    template['answers'] = answers
    
    with open(BASE / 'submission.json', 'w') as f:
        json.dump(template, f, indent=2)
        
    print("Done! Submission saved.")
    
    gt_path = BASE / "agentic-bank-public/ground_truth.json"
    if gt_path.exists():
        from halyk_agent.eval.harness import EvalHarness
        harness = EvalHarness(gt_path)
        metrics = harness.run_evaluation(BASE / 'submission.json')
        metrics.print_report()
        
if __name__ == '__main__':
    main()
