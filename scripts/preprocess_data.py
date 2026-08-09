"""
STEP 1: Pre-process all documents and extract structured covenant definitions.
This creates data/preprocessed/ with all needed data for fast runtime.

Output:
  data/preprocessed/covenants.json - covenant definitions per scenario
  data/preprocessed/doc_texts.json - extracted text per document 
  data/preprocessed/affiliates.json - known affiliated parties per scenario
"""
import fitz
import json
import re
import os
from pathlib import Path

BASE = Path('E:/AntigravityProjects/halyk-agent')
DOCS_DIR = BASE / 'agentic-bank-public/documents'
OUTPUT_DIR = BASE / 'data/preprocessed'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(BASE / 'data/scenario_mapping.json') as f:
    mapping = json.load(f)

# ============================================================
# 1. Extract text from ALL PDFs
# ============================================================
print("=== STEP 1: Extract text from all PDFs ===")
doc_texts = {}
for pdf_file in DOCS_DIR.glob('*.pdf'):
    doc_id = pdf_file.stem
    try:
        with fitz.open(pdf_file) as doc:
            text = ''
            for page in doc:
                text += page.get_text('text', flags=fitz.TEXT_PRESERVE_WHITESPACE)
        doc_texts[doc_id] = text
        print(f'  {doc_id}: {len(text)} chars')
    except Exception as e:
        print(f'  {doc_id}: ERROR {e}')

with open(OUTPUT_DIR / 'doc_texts.json', 'w', encoding='utf-8') as f:
    json.dump(doc_texts, f, ensure_ascii=False)
print(f'Saved {len(doc_texts)} documents\n')

# ============================================================
# 2. Identify doc types using metadata cache or heuristics 
# ============================================================
print("=== STEP 2: Classify documents ===")
doc_metas = {}
for doc_id, text in doc_texts.items():
    # Check cached metadata first
    meta_path = DOCS_DIR / f'{doc_id}_meta.json'
    if meta_path.exists():
        with open(meta_path) as f:
            doc_metas[doc_id] = json.load(f)
        continue
    
    # Heuristic classification
    text_lower = text.lower()
    meta = {
        'doc_type': 'OTHER',
        'valid_from': None,
        'valid_to': None,
        'amends_contract_date': None
    }
    
    if '6.1' in text and '6.2' in text and '6.3' in text:
        meta['doc_type'] = 'CONTRACT'
        # Try to find validity dates
        date_match = re.search(r'20(\d{2})-0[1]?-01', text)
        if date_match:
            year = int('20' + date_match.group(1))
            meta['valid_from'] = f'{year}-01-01'
    elif 'kyc' in text_lower or 'know your customer' in text_lower or 'beneficial owner' in text_lower:
        meta['doc_type'] = 'KYC'
    elif 'audit' in text_lower or 'financial statement' in text_lower or 'balance sheet' in text_lower:
        meta['doc_type'] = 'AUDIT'
    elif 'amendment' in text_lower or 'addendum' in text_lower:
        meta['doc_type'] = 'AMENDMENT'
    
    doc_metas[doc_id] = meta
    # Save cache
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False)

with open(OUTPUT_DIR / 'doc_metas.json', 'w', encoding='utf-8') as f:
    json.dump(doc_metas, f, ensure_ascii=False, indent=2)
print(f'Classified {len(doc_metas)} documents\n')

# ============================================================
# 3. Extract covenant definitions from contracts
# ============================================================
print("=== STEP 3: Extract covenant definitions ===")

def extract_covenants_from_contract(text, doc_id):
    """Extract covenant 6.1, 6.2, 6.3 definitions from contract text."""
    covenants = {}
    
    # Find validity period
    valid_from = None
    date_matches = re.findall(r'(20\d{2}-\d{2}-\d{2})\s*[^0-9]*?(20\d{2}-\d{2}-\d{2})', text)
    if date_matches:
        valid_from = date_matches[0][0]
    
    for cov_num in ['6.1', '6.2', '6.3']:
        # Get text around the covenant
        m = re.search(rf'({re.escape(cov_num)}.*?)(?=\b6\.[{int(cov_num[-1])+1}9]\b|\bSection\s*7|\b§\s*7|$)', text, re.DOTALL)
        if not m:
            continue
        
        snippet = m.group(1)[:800]
        
        cov_def = {
            'covenant_num': cov_num,
            'valid_from': valid_from,
            'snippet': snippet[:800],
        }
        
        # Determine type: RATIO or DOLLAR LIMIT
        ratio_match = re.search(r'(\d+\.?\d*)\s*x', snippet)
        dollar_match = re.search(r'\$([0-9,]+\.?\d*)', snippet)
        
        # Determine what KIND of covenant it is from the snippet
        snippet_lower = snippet.lower().replace('\n', ' ')
        
        # Identify covenant type by keywords
        if 'related.?party' in re.sub(r'\s+', '', snippet_lower) or 'related party' in snippet_lower:
            cov_def['category'] = 'related_party'
        elif 'capital intensity' in snippet_lower or 'capex' in snippet_lower:
            cov_def['category'] = 'capex_ratio'  # CAPEX / Revenue
        elif 'ebitda' in snippet_lower and ('debt' in snippet_lower or 'net' in snippet_lower):
            cov_def['category'] = 'debt_ebitda_ratio'
        elif 'ebitda' in snippet_lower:
            cov_def['category'] = 'ebitda_ratio'
        elif 'overhead' in snippet_lower or 'individual' in snippet_lower:
            cov_def['category'] = 'individual_overhead'
        elif 'cover' in snippet_lower and 'application' in snippet_lower:
            cov_def['category'] = 'coverage_ratio'
        elif 'revenue' in snippet_lower and ('minimum' in snippet_lower or 'min' in snippet_lower):
            cov_def['category'] = 'min_revenue'
        elif 'revenue' in snippet_lower:
            cov_def['category'] = 'revenue_limit'
        elif 'debt' in snippet_lower or 'loan' in snippet_lower or 'borrowing' in snippet_lower:
            cov_def['category'] = 'debt_limit'
        elif 'capex' in snippet_lower or 'capital' in snippet_lower:
            cov_def['category'] = 'capex_limit'
        else:
            cov_def['category'] = 'unknown'
        
        if ratio_match:
            cov_def['limit_type'] = 'RATIO'
            cov_def['limit_value'] = float(ratio_match.group(1))
        elif dollar_match:
            cov_def['limit_type'] = 'MAX'
            cov_def['limit_value'] = float(dollar_match.group(1).replace(',', ''))
        else:
            cov_def['limit_type'] = 'UNKNOWN'
            cov_def['limit_value'] = None
        
        # Check for "proportion of revenue" pattern
        if 'proportion of revenue' in snippet_lower or 'as a proportion' in snippet_lower:
            cov_def['limit_type'] = 'RATIO'
            cov_def['category'] = 'related_party_ratio'
        
        covenants[cov_num] = cov_def
    
    return covenants

scenario_covenants = {}
for sid in sorted(mapping.keys()):
    docs = mapping[sid]
    contracts = []
    
    for doc_id in docs:
        meta = doc_metas.get(doc_id, {})
        if meta.get('doc_type') == 'CONTRACT':
            covs = extract_covenants_from_contract(doc_texts.get(doc_id, ''), doc_id)
            contracts.append({
                'doc_id': doc_id,
                'valid_from': meta.get('valid_from'),
                'covenants': covs
            })
    
    # Pick the contract that covers 2025 (prefer valid_from=2025-01-01)
    best_contract = None
    for c in contracts:
        vf = c.get('valid_from', '')
        if vf and '2025' in vf:
            best_contract = c
            break
    if not best_contract and contracts:
        # Fallback to latest
        contracts.sort(key=lambda x: x.get('valid_from', '') or '', reverse=True)
        best_contract = contracts[0]
    
    if best_contract:
        scenario_covenants[sid] = {
            'contract_doc_id': best_contract['doc_id'],
            'valid_from': best_contract['valid_from'],
            'covenants': best_contract['covenants']
        }
        print(f'  {sid}: contract={best_contract["doc_id"]}, valid_from={best_contract["valid_from"]}')
        for cov_id, cov_def in best_contract['covenants'].items():
            print(f'    {cov_id}: {cov_def["category"]} | {cov_def["limit_type"]}={cov_def.get("limit_value")}')
    else:
        print(f'  {sid}: NO CONTRACT FOUND!')

with open(OUTPUT_DIR / 'covenants.json', 'w', encoding='utf-8') as f:
    json.dump(scenario_covenants, f, ensure_ascii=False, indent=2)
print(f'\nSaved covenant definitions for {len(scenario_covenants)} scenarios')

# ============================================================
# 4. Extract affiliated parties from KYC documents
# ============================================================
print("\n=== STEP 4: Extract affiliated parties ===")
scenario_affiliates = {}
for sid in sorted(mapping.keys()):
    docs = mapping[sid]
    affiliates = []
    
    for doc_id in docs:
        meta = doc_metas.get(doc_id, {})
        if meta.get('doc_type') == 'KYC':
            text = doc_texts.get(doc_id, '')
            # Extract company/entity names from KYC
            # Look for patterns like "Related party: X" or "Affiliate: X"
            # Also look for LLP, JSC, LLC patterns
            entities = re.findall(r'(?:related.?party|affiliate|benefic|shareholder|parent.?company|subsidiary|associated.?company)[:\s]+([A-Z][a-zA-Z\s,]+(?:LLP|JSC|LLC|Corp|Ltd|Inc|L\.L\.P\.))', text, re.IGNORECASE)
            affiliates.extend(entities)
            
            # Also find Kazakh-style company names
            kazakh_entities = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:LLP|JSC|L\.L\.P\.))\b', text)
            affiliates.extend(kazakh_entities)
    
    if affiliates:
        # Deduplicate
        affiliates = list(set(affiliates))
        scenario_affiliates[sid] = affiliates
        print(f'  {sid}: {len(affiliates)} affiliates: {affiliates[:5]}')

with open(OUTPUT_DIR / 'affiliates.json', 'w', encoding='utf-8') as f:
    json.dump(scenario_affiliates, f, ensure_ascii=False, indent=2)

print(f'\nPreprocessing complete! Files saved to {OUTPUT_DIR}')
