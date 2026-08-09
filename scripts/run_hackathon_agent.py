from pathlib import Path
import os
import sys
import codecs
import json
from loguru import logger
import pandas as pd
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field

# Fix Windows console encoding for easyocr progress bars
if sys.platform == "win32":
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "replace")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "replace")

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from halyk_agent.graph.kyc_graph import init_kuzu_db, extract_and_store_affiliates, get_affiliates_for_scenario
from halyk_agent.utils.llm_factory import LLMFactory
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from halyk_agent.eval.harness import EvalHarness
from tenacity import retry, stop_after_attempt, wait_exponential
from rank_bm25 import BM25Okapi
import re
import threading

class CovenantDefinition(BaseModel):
    description: str = Field(description="Full description of the covenant logic in English")
    limit_value: Optional[float] = Field(description="Numeric limit value (e.g. 300000.0). Null if complex ratio without specific sum")
    limit_type: Literal["MAX", "MIN", "RATIO"] = Field(description="Limit type: MAX (not greater), MIN (not less), RATIO (coefficient)")
    transaction_category: str = Field(description="Transaction category IN ENGLISH EXACTLY as it appears in the ledger (e.g. Net Debt, Net Loss). For RATIO, this is the numerator.")
    transaction_category_denominator: Optional[str] = Field(description="Denominator category IN ENGLISH EXACTLY as it appears in the ledger (e.g. EBITDA). Return null if not a RATIO.")
    exchange_rates: Optional[dict] = Field(default=None, description="Dictionary of fixed exchange rates to USD if explicitly stated in text (e.g. {'EUR': 1.05}). Null otherwise.")
    is_single_transaction: bool = Field(description="True if limit applies to each single transaction. False if applies to sum of all transactions over period.")

class DocMetadata(BaseModel):
    doc_type: str = Field(description="One of: CONTRACT, AMENDMENT, KYC, AUDIT, OTHER")
    valid_from: Optional[str] = Field(description="Start date of validity (YYYY-MM-DD), or null if not stated")
    valid_to: Optional[str] = Field(description="End date of validity (YYYY-MM-DD), or null if indefinitely valid or not stated")
    amends_contract_date: Optional[str] = Field(description="If this is an AMENDMENT, what is the date of the original contract it amends (YYYY-MM-DD)? Null otherwise.")


class TransactionMatch(BaseModel):
    matched_txn_ids: List[str] = Field(description="Список ID транзакций, которые относятся к данной категории")

class TxnMatchItem(BaseModel):
    txn_id: str = Field(description="The transaction ID")
    match_status: str = Field(description="Strictly 'MATCH' if it belongs to category, or 'IGNORE' otherwise")

class BatchClassification(BaseModel):
    results: List[TxnMatchItem] = Field(description="Classification for each transaction")


COVENANT_CACHE_PATH = Path("data/covenant_cache.json")
CACHE_LOCK = threading.Lock()

def robust_json_parse(text: str) -> dict:
    """Bulletproof JSON parser for Reasoning models that strips <think> tags."""
    # Remove <think> blocks if present
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Extract everything from the first { to the last }
    match = re.search(r'(\{.*\})', text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in output")
        
    json_str = match.group(1)
    
    # Clean up markdown code blocks if the model ignored instructions
    json_str = json_str.strip()
    if json_str.startswith("```json"):
        json_str = json_str[7:]
    if json_str.startswith("```"):
        json_str = json_str[3:]
    if json_str.endswith("```"):
        json_str = json_str[:-3]
        
    try:
        return json.loads(json_str.strip())
    except json.JSONDecodeError:
        import json_repair
        return json_repair.repair_json(json_str, return_objects=True)

def load_cache():
    with CACHE_LOCK:
        if COVENANT_CACHE_PATH.exists():
            try:
                with open(COVENANT_CACHE_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return {k: CovenantDefinition(**v) for k, v in data.items()}
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")
        return {}

def save_cache(cache):
    with CACHE_LOCK:
        try:
            COVENANT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(COVENANT_CACHE_PATH, "w", encoding="utf-8") as f:
                json_data = {k: v.model_dump() for k, v in cache.items()}
                json.dump(json_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

COVENANT_CACHE = load_cache()

def prepare_doc_text_for_extraction(doc_texts: List[str], covenant_num: str) -> str:
    """
    Prepare document text for covenant extraction.
    Only pass snippet from CONTRACT documents that mention the covenant, plus the first 1500 chars (for validity dates).
    """
    parts = []
    
    for i, doc_text in enumerate(doc_texts):
        if "[METADATA: Type=CONTRACT" not in doc_text:
            continue
            
        covenant_match = re.search(rf'\b{re.escape(covenant_num)}\b', doc_text)
        if covenant_match:
            metadata_header = doc_text[:1500] 
            
            start_idx = max(0, covenant_match.start() - 2500)
            end_idx = min(len(doc_text), covenant_match.end() + 2500)
            snippet = doc_text[start_idx:end_idx]
            
            if start_idx < 1500:
                parts.append(f"--- CONTRACT DOCUMENT {i+1} ---\n{doc_text[:end_idx]}\n")
            else:
                parts.append(f"--- CONTRACT DOCUMENT {i+1} ---\n{metadata_header}\n[... snipped ...]\n{snippet}\n")
    
    if not parts:
        for i, doc_text in enumerate(doc_texts):
            if "[METADATA: Type=CONTRACT" in doc_text:
                parts.append(f"--- CONTRACT DOCUMENT {i+1} ---\n{doc_text[:2500]}\n[... snipped ...]")
                
    final_text = "\n\n".join(parts)
    
    # Save debug
    from pathlib import Path
    debug_dir = Path("debug_runs")
    debug_dir.mkdir(exist_ok=True)
    with open(debug_dir / f"extract_{covenant_num}_input.txt", "w", encoding="utf-8") as f:
        f.write(final_text)
    
    logger.info(f"Prepared {len(final_text)} chars for extraction (covenant {covenant_num})")
    return final_text

METADATA_CACHE_PATH = Path("data/metadata_cache.json")

def load_metadata_cache() -> Dict[str, DocMetadata]:
    if METADATA_CACHE_PATH.exists():
        try:
            with open(METADATA_CACHE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {k: DocMetadata(**v) for k, v in data.items()}
        except Exception as e:
            logger.warning(f"Failed to load metadata cache: {e}")
    return {}

METADATA_CACHE = load_metadata_cache()

@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=1, max=10), retry_error_callback=lambda retry_state: None)
def analyze_document_temporal_metadata(llm, text: str, doc_id: str) -> Optional[DocMetadata]:
    # Get a fresh LLM on every try to seamlessly rotate API keys
    fresh_llm = LLMFactory.get_llm()
    
    global METADATA_CACHE
    METADATA_CACHE = load_metadata_cache()
    
    if doc_id in METADATA_CACHE:
        logger.info(f"Using cached metadata for doc {doc_id}")
        return METADATA_CACHE[doc_id]
        
    head = text[:3000]
    tail = text[-1000:] if len(text) > 4000 else ""
    content = head + "\n\n... [MIDDLE REMOVED] ...\n\n" + tail
    
    parser = JsonOutputParser(pydantic_object=DocMetadata)
    prompt = PromptTemplate(
        template='''You are a legal document analyzer. Extract metadata from the following document.
        
DOCUMENT TEXT:
{content}

Analyze the document and provide the metadata in JSON format.
For dates, use YYYY-MM-DD format. If a contract is valid "until terminated", valid_to is null.
Pay special attention to whether this is a Master Agreement (CONTRACT) or an Addendum/Amendment (AMENDMENT).

{format_instructions}
''',
        input_variables=["content"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    formatted_prompt = prompt.format(content=content)
    structured_llm = fresh_llm.with_structured_output(DocMetadata)
    res_obj = structured_llm.invoke(formatted_prompt)
    if not res_obj:
        raise ValueError("LLM returned None instead of DocMetadata")
    
    # Save cache
    METADATA_CACHE[doc_id] = res_obj
    METADATA_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(METADATA_CACHE_PATH, "w", encoding="utf-8") as f:
            json_data = {k: v.model_dump() for k, v in METADATA_CACHE.items()}
            json.dump(json_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save metadata cache: {e}")
        
    return res_obj

@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=1, max=10), retry_error_callback=lambda retry_state: None)
def extract_covenant_from_doc(llm, doc_texts: List[str], covenant_num: str, target_years: str, scenario_id: str = "") -> Optional[CovenantDefinition]:
    full_text = prepare_doc_text_for_extraction(doc_texts, covenant_num)
    import hashlib
    content_hash = hashlib.md5(full_text.encode('utf-8')).hexdigest()
    cache_key = f"{scenario_id}_{covenant_num}"
    
    # Reload cache to sync with other parallel workers
    global COVENANT_CACHE
    COVENANT_CACHE = load_cache()
    
    if cache_key in COVENANT_CACHE:
        logger.info(f"Using cached definition for covenant {covenant_num} (Years: {target_years})")
        return COVENANT_CACHE[cache_key]
        
    # Removed faulty fast path to ensure LLM parses category correctly

    parser = JsonOutputParser(pydantic_object=CovenantDefinition)
    prompt = PromptTemplate(
        template='''You are a financial analyst. Read the loan agreement documents below and extract covenant (clause) {covenant_num}.

RULES FOR DETERMINING THE ACTIVE AGREEMENT:
- Transactions in the ledger for this scenario occur in: {target_years}.
- Each contract or amendment has a validity period. Extract limits ONLY from the document whose validity period covers {target_years}!
- STRICTLY IGNORE any stamps, watermarks, or marks like "INACTIVE VERSION". They have no legal force. Look ONLY at the start and end dates in the text itself!

COVENANT TYPE HEURISTICS:
- If a covenant describes a ratio (like DSCR or CAPEX/OPEX), set limit_type to 'RATIO', and fill both transaction_category and transaction_category_denominator.
- If a covenant is a maximum sum allowed (e.g. Total Debt, Total CAPEX), set limit_type to 'MAX' and denominator to null.
- If a covenant is a minimum sum required (e.g. Minimum Revenue), set limit_type to 'MIN' and denominator to null.
- If a covenant limits a single transaction rather than an aggregate sum (e.g. single purchase of equipment), set is_single_transaction to True.

CRITICAL INSTRUCTIONS:
- Match the exact covenant number ({covenant_num}) to the text! You can find the numbers like "6.1", "6.2", "6.3" in the contract text. Even if the text is long, find exactly where covenant {covenant_num} is defined.
- For transaction_category and transaction_category_denominator, write a DETAILED description of what transactions belong to this category. Include the EXACT definition from the contract text if available (e.g. "Operating Expenses as defined in Section 1: means all costs directly related to...").
- For is_single_transaction: set True ONLY if the covenant tests each transaction individually against the limit. Set False if it tests the aggregate sum.
- EXCHANGE RATES: If the text explicitly states a conversion rate (e.g. 1 EUR = 1.05 USD), extract it into exchange_rates. Otherwise null.

Documents:
{doc_text}

Return JSON:
{format_instructions}
''',
        input_variables=["covenant_num", "doc_text", "target_years"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    try:
        formatted_prompt = prompt.format(covenant_num=covenant_num, doc_text=full_text, target_years=target_years)
        fresh_llm = LLMFactory.get_llm()
        structured_llm = fresh_llm.with_structured_output(CovenantDefinition)
        res_obj = structured_llm.invoke(formatted_prompt)
        if not res_obj:
            raise ValueError("LLM returned None instead of CovenantDefinition")
        
        # Save debug
        debug_dir = Path("debug_runs")
        debug_dir.mkdir(exist_ok=True)
        with open(debug_dir / f"extract_{covenant_num}_response.txt", "w", encoding="utf-8") as f:
            f.write(str(res_obj.model_dump()))
            
        COVENANT_CACHE[cache_key] = res_obj
        save_cache(COVENANT_CACHE)  # Persistent save
        return res_obj
    except Exception as e:
        logger.error(f"Extract covenant {covenant_num} failed: {e}")
        raise e
def filter_transactions_for_covenant(txns, category, snippet="", affiliates=None):
    if len(txns) == 0:
        return txns

    category = str(category).lower()

    # Heuristic mapping for standard hackathon categories
    if "related part" in category or "affiliate" in category:
        if affiliates:
            pattern = "|".join([re.escape(a.lower()) for a in affiliates])
            mask = txns["counterparty"].str.lower().str.contains(pattern, na=False)
            return txns[mask]
        else:
            return txns

    # Filter by keywords
    desc_series = txns["description"].str.lower()
    cp_series = txns["counterparty"].str.lower().fillna("")
    ex_mask = desc_series.str.contains('interest|\u043f\u0440\u043e\u0446\u0435\u043d\u0442|tax|\u043d\u0430\u043b\u043e\u0433|depreciation|amortization', regex=True) | \
              cp_series.str.contains('interest|\u043f\u0440\u043e\u0446\u0435\u043d\u0442|tax|\u043d\u0430\u043b\u043e\u0433|depreciation|amortization', regex=True)

    if 'revenue' in category or 'sales' in category:
        mask = desc_series.str.contains('revenue|sales|income|\u0432\u044b\u0440\u0443\u0447|\u0434\u043e\u0445\u043e\u0434', regex=True)
        ex_mask = ex_mask | desc_series.str.contains('refund|rebate|credit|return|recovery|sweep|advance', regex=True)
        matched = txns[mask & ~ex_mask & (txns["amount"] > 0)]

        # Check if covenant specifies Q4 only
        s = snippet.lower() if snippet else ""
        if '2025-12-31' in s and '2025-01-01' not in s:
            # Filter for Q4 (months 10, 11, 12)
            matched = matched[pd.to_datetime(matched['date']).dt.month >= 10]
        elif '2025-09-30' in s and '2025-01-01' not in s:
            matched = matched[pd.to_datetime(matched['date']).dt.month >= 7]

        return matched

    elif "capex" in category or "capital" in category:
        mask = desc_series.str.contains('capex|capital|equipment|asset|construction|\u043a\u0430\u043f\u0438\u0442\u0430\u043b\u044c|purchase', regex=True)
        capex_exclude = desc_series.str.contains('lease|rent|\u0430\u0440\u0435\u043d\u0434', regex=True)
        return txns[mask & ~ex_mask & ~capex_exclude & (txns["amount"] < 0)]

    elif "opex" in category or "operating" in category:
        mask = desc_series.str.contains('opex|operating|salary|rent|utility|insurance|fee|cost|\u0437\u0430\u0440\u0430\u0431\u043e\u0442\u043d\u0430\u044f|marketing|payroll', regex=True)
        return txns[mask & ~ex_mask & (txns["amount"] < 0)]
        
    elif "debt" in category or "loan" in category or "borrowing" in category:
        mask = desc_series.str.contains('debt|loan|borrowing|revolver|credit|principal|займ|кредит', regex=True)
        return txns[mask]
        
    elif "interest" in category:
        mask = desc_series.str.contains('interest|процент', regex=True)
        return txns[mask]
        
    elif "ebitda" in category:
        # EBITDA is Revenue + OPEX
        mask = desc_series.str.contains('revenue|sales|income|received|settlement|\u0432\u044b\u0440\u0443\u0447\u043a\u0430|opex|operating|salary|rent|utility|insurance|fee|cost|\u0437\u0430\u0440\u0430\u0431\u043e\u0442\u043d\u0430\u044f|marketing|payroll', regex=True)
        ex_mask = desc_series.str.contains('advance payment|prepayment|unearned', regex=True)
        return txns[mask & ~ex_mask]
        
    elif "tax" in category:
        mask = desc_series.str.contains('tax|налог', regex=True)
        return txns[mask]
        
    elif "dividend" in category:
        mask = desc_series.str.contains('dividend|дивиденд', regex=True)
        return txns[mask]
        
    else:
        # Fallback to definition words
        words = category.replace(",", " ").split()
        keywords = [re.escape(k) for k in words if len(k) > 3]
        if not keywords:
            return txns
        pattern = "|".join(keywords)
        mask = desc_series.str.contains(pattern, regex=True)
        return txns[mask]

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
def llm_batch_classify(llm, category_def: str, txns_text: str) -> BatchClassification:
    # Get a fresh LLM on every try to seamlessly rotate API keys if one gets rate-limited
    fresh_llm = LLMFactory.get_llm()
    
    parser = JsonOutputParser(pydantic_object=BatchClassification)
    prompt = PromptTemplate(
        template='''You are a precise financial auditor. Given a strict CATEGORY DEFINITION and a list of TRANSACTIONS, classify EACH transaction.

CATEGORY DEFINITION:
{category_def}

TRANSACTIONS:
{txns_text}

INSTRUCTIONS:
1. Read the category definition carefully, paying attention to what is included and excluded.
2. For each transaction, decide if it belongs to the category.
3. Output a JSON object with a list called 'results'.
4. For EACH transaction, provide its 'txn_id' and 'match_status' ("MATCH" or "IGNORE").

{format_instructions}
''',
        input_variables=["category_def", "txns_text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    formatted_prompt = prompt.format(category_def=category_def, txns_text=txns_text)
    structured_llm = fresh_llm.with_structured_output(BatchClassification)
    res_obj = structured_llm.invoke(formatted_prompt)
    if not res_obj:
        raise ValueError("LLM returned None instead of BatchClassification")
    return res_obj

def filter_transactions_with_llm(txns: pd.DataFrame, category_def: str, llm, snippet: str = "", affiliates: List[str] = None) -> pd.DataFrame:
    if txns.empty:
        return txns

    # Heuristic for related party - it's 100% deterministic based on affiliate names
    cat_lower = str(category_def).lower()
    if "related part" in cat_lower or "affiliate" in cat_lower:
        if affiliates:
            pattern = "|".join([re.escape(a.lower()) for a in affiliates])
            mask = txns["counterparty"].str.lower().str.contains(pattern, na=False)
            return txns[mask]
        return txns # No affiliates found
        
    # Check if covenant specifies Q4 or Q3 only for the period
    s = snippet.lower() if snippet else ""
    date_filtered_txns = txns.copy()
    if '2025-12-31' in s and '2025-01-01' not in s:
        date_filtered_txns = txns[pd.to_datetime(txns['date']).dt.month >= 10]
    elif '2025-09-30' in s and '2025-01-01' not in s:
        date_filtered_txns = txns[pd.to_datetime(txns['date']).dt.month >= 7]

    tx_records = date_filtered_txns.to_dict('records')
    matched_ids = set()
    
    chunk_size = 35  # send up to 35 transactions at once to avoid token limits
    import math
    
    for i in range(0, len(tx_records), chunk_size):
        chunk = tx_records[i:i+chunk_size]
        txns_text = ""
        for t in chunk:
            txns_text += f"ID: {t['txn_id']} | Date: {t['date']} | Amount: {t['amount']} | Counterparty: {t['counterparty']} | Purpose: {t['description']}\n"
            
        try:
            res = llm_batch_classify(llm, category_def, txns_text)
            for item in res.results:
                if item.match_status == "MATCH":
                    matched_ids.add(item.txn_id)
        except Exception as e:
            logger.error(f"Batch classification failed: {e}")
            # Fallback to heuristic for this chunk
            chunk_df = pd.DataFrame(chunk)
            fallback_df = filter_transactions_for_covenant(chunk_df, category_def, snippet, affiliates)
            matched_ids.update(fallback_df['txn_id'].tolist())
            
    # return DataFrame with matched IDs
    if matched_ids:
        return txns[txns['txn_id'].isin(matched_ids)]
    else:
        return txns.iloc[0:0] # return empty dataframe if nothing matched



def process_scenario(scenario_id: str, account_id: str, doc_ids: List[str], ledger_df: pd.DataFrame, template_covenants: dict, raw_dir: Path, llm, current_idx: int, total_scenarios: int) -> dict:
    logger.info(f"Processing scenario {scenario_id} for account {account_id} ({current_idx} of {total_scenarios})")
    
    # Load affiliates directly from preprocessed json
    affiliates = []
    affiliates_path = raw_dir.parent / "data/preprocessed/affiliates.json"
    if affiliates_path.exists():
        with open(affiliates_path, "r", encoding="utf-8") as f:
            all_aff = json.load(f)
            affiliates = all_aff.get(scenario_id, [])

    
    # Get all transactions for this scenario first to know the target year
    scenario_txns = ledger_df[ledger_df['txn_id'].str.startswith(f"TXN-{scenario_id}-")]
    tx_records = scenario_txns.to_dict('records')
    years = sorted(list(set([str(d)[:4] for d in scenario_txns['date']]))) if not scenario_txns.empty else ["2025"]
    target_years = ", ".join(years)
    target_year = int(years[0])
    
    # Load document texts and metadata
    doc_texts_dict = {}
    doc_metas = {}
    import fitz
    for doc_id in doc_ids:
        pdf_name = doc_id if doc_id.endswith('.pdf') else f"{doc_id}.pdf"
        pdf_path = raw_dir / pdf_name
        if pdf_path.exists():
            try:
                with fitz.open(pdf_path) as doc:
                    text = ""
                    pages_with_covenants = set([0, 1]) # always OCR first 2 pages for metadata
                    for i, page in enumerate(doc):
                        page_text = page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE)
                        text += f"--- Page {i+1} ---\n{page_text}\n"
                        # even if garbled, numbers like 6.1, 6.2, 6.3 survive
                        if "6.1" in page_text or "6.2" in page_text or "6.3" in page_text or "6." in page_text:
                            pages_with_covenants.add(i)
                            # also add adjacent pages just in case it spans across pages
                            pages_with_covenants.add(max(0, i-1))
                            pages_with_covenants.add(min(len(doc)-1, i+1))
                    
                    garbled_count = text.count('\uFFFD')
                    if len(text.strip()) < 1000 or garbled_count > len(text) * 0.05:
                        logger.warning(f"  Doc {doc_id} needs OCR... OCRing {len(pages_with_covenants)} pages instead of full doc.")
                        from halyk_agent.utils.ocr_engine import extract_text_from_pdf_ocr
                        ocr_text = extract_text_from_pdf_ocr(str(pdf_path), page_numbers=list(pages_with_covenants))
                        if len(ocr_text) > 100:
                            text = ocr_text
                            
                    doc_texts_dict[doc_id] = text
                
                # Try cache first? No, for hackathon we can just run it, but let's cache temporal metadata locally
                meta_cache_path = raw_dir / f"{doc_id}_meta.json"
                if meta_cache_path.exists():
                    with open(meta_cache_path, "r", encoding="utf-8") as f:
                        meta_dict = json.load(f)
                        meta = DocMetadata(**meta_dict)
                else:
                    meta = analyze_document_temporal_metadata(llm, text, doc_id)
                    if meta:
                        with open(meta_cache_path, "w", encoding="utf-8") as f:
                            json.dump(meta.model_dump(), f, ensure_ascii=False)
                            
                if meta:
                    doc_metas[doc_id] = meta
                    logger.info(f"  Doc {doc_id} -> {meta.doc_type} | {meta.valid_from} to {meta.valid_to}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"Failed to read/analyze {pdf_path}: {e}")
                
    # Filter documents chronologically
    logger.info(f"  DEBUG 2: len(doc_texts_dict) = {len(doc_texts_dict)}, len(doc_ids) = {len(doc_ids)}")
    valid_docs = []
    
    for doc_id, text in doc_texts_dict.items():
        meta = doc_metas.get(doc_id)
        if meta and meta.valid_to:
            try:
                valid_to_year = int(meta.valid_to[:4])
                if valid_to_year < target_year:
                    logger.warning(f"  Skipping doc {doc_id}: expired in {valid_to_year} (target: {target_year})")
                    continue
            except:
                pass
        
        if meta and meta.valid_from:
            try:
                valid_from_year = int(meta.valid_from[:4])
                if valid_from_year > int(years[-1]):
                    logger.warning(f"  Skipping doc {doc_id}: future doc from {valid_from_year} (target: {years[-1]})")
                    continue
            except:
                pass
                
        header = ""
        if meta:
            header = f"\n[METADATA: Type={meta.doc_type}, Valid_from={meta.valid_from}, Valid_to={meta.valid_to}]\n"
        valid_docs.append(header + text)
        
    doc_texts = valid_docs
    

    
    answers = {}

    # Build audit context from documents marked as AUDIT or OTHER
    
    audit_texts = []
    for d_id, meta in doc_metas.items():
        if meta and meta.doc_type == "KYC":
            continue

        if meta and (meta.doc_type == "AUDIT" or meta.doc_type == "OTHER"):
            text = doc_texts_dict.get(d_id, "")
            if len(text) > 200:
                audit_texts.append(f"Document {d_id}:\n" + text[:2000])

    aff_text = "\n".join(affiliates)
    audit_context = "\n\n---\n\n".join(audit_texts)
    if affiliates:
        audit_context = f"AFFILIATED PARTIES IDENTIFIED FROM KYC (STRICT ALLOWLIST):\n{aff_text}\n\n---\n\n" + audit_context

    debug_data = {
        "scenario_id": scenario_id,
        "account_id": account_id,
        "covenants": {}
    }
    
    for cov_id, cov_data in template_covenants.items():
        snippet = cov_data.get("snippet", "")
        logger.info(f"  Analyzing Covenant {cov_id}")
        ans = {"status": "COMPLIANT", "actual": 0.0, "evidence_txn_id": None}
        matched_txns = []
        den_matched_txns = []
        
        logger.info(f"  DEBUG: len(doc_texts) = {len(doc_texts)}, len(valid_docs) = {len(valid_docs)}")
        
        if not doc_texts:
            logger.warning("  No document text found!")
            answers[cov_id] = ans
            continue
            
        try:
            cov_def = extract_covenant_from_doc(llm, doc_texts, cov_id, target_years, scenario_id)
        except Exception as e:
            logger.error(f"  Extract covenant {cov_id} failed: {e}")
            cov_def = None
            
        if not cov_def:
            answers[cov_id] = ans
            continue
            
        logger.info(f"    Definition: {cov_def.transaction_category} | Limit: {cov_def.limit_value} ({cov_def.limit_type})")
        
        filtered_txns_df = filter_transactions_with_llm(scenario_txns, cov_def.transaction_category, llm, snippet, affiliates)
        tx_records = filtered_txns_df.to_dict('records')
        
        matched_txns = tx_records
        logger.info(f"    Matched {len(matched_txns)} transactions using LLM batch")
        
        # Accumulate metrics
        breach_found = False
        num_total = 0.0
        converted_txns_amt = []
        for tx in matched_txns:
            amt = float(tx['amount'])
            
            # If this is a related party payments covenant (usually MAX limit for payments), we only care about expenses
            if "related part" in str(cov_def.transaction_category).lower() and amt > 0:
                continue # Skip positive amounts (revenue) for related party payments
                
            curr = str(tx.get('currency', 'USD')).upper()
            if curr != 'USD':
                if cov_def.exchange_rates and curr in cov_def.exchange_rates:
                    amt = amt * cov_def.exchange_rates[curr]
                else:
                    logger.warning(f"    WARNING: No exchange rate for {curr} in cov {cov_id}! Using 1:1.")
            converted_txns_amt.append((tx['txn_id'], amt))
            
        if cov_def.is_single_transaction:
            # Check individual transactions
            breach_found = False
            for tid, amt in converted_txns_amt:
                val = abs(amt)
                if cov_def.limit_type == 'MAX' and val > (cov_def.limit_value or float('inf')):
                    ans["status"] = "BREACH"
                    ans["actual"] = round(val, 2)
                    ans["evidence_txn_id"] = tid
                    breach_found = True
                    break    
            if not breach_found:
                ans["status"] = "COMPLIANT"
                ans["actual"] = max([abs(float(t['amount'])) for t in matched_txns]) if matched_txns else 0.0
                
        else:
            is_ratio = (cov_id == '6.1' or cov_def.limit_type == 'RATIO' or bool(cov_def.transaction_category_denominator))
            
            if is_ratio:
                num_total = abs(sum([amt for _, amt in converted_txns_amt]))
                
                # Denominator processing
                den_total = 1.0
                den_converted_amt = []
                if cov_def.transaction_category_denominator:
                    cov_def_den = CovenantDefinition(
                        description=cov_def.description,
                        limit_value=cov_def.limit_value,
                        limit_type=cov_def.limit_type,
                        transaction_category=cov_def.transaction_category_denominator,
                        transaction_category_denominator=None,
                        is_single_transaction=False,
                        exchange_rates=cov_def.exchange_rates
                    )
                    den_filtered_df = filter_transactions_with_llm(scenario_txns, cov_def_den.transaction_category, llm, snippet, affiliates)
                    den_tx_records = den_filtered_df.to_dict('records')
                    den_matched_txns = den_tx_records
                    
                    for tx in den_matched_txns:
                        amt = float(tx['amount'])
                        curr = str(tx.get('currency', 'USD')).upper()
                        if curr != 'USD' and cov_def_den.exchange_rates and curr in cov_def_den.exchange_rates:
                            amt = amt * cov_def_den.exchange_rates[curr]
                        den_converted_amt.append((tx['txn_id'], amt))
                        
                    den_total = abs(sum([a for _, a in den_converted_amt]))
                    if den_total == 0:
                        den_total = 1.0 # prevent div zero
                        
                actual = round(num_total / den_total, 4)
                ans["actual"] = actual
                ans["evidence_txn_id"] = None
                
                # RATIO breaches
                if cov_def.limit_type == 'MIN':
                    ans["status"] = "BREACH" if actual < (cov_def.limit_value or 0.0) else "COMPLIANT"
                else: # Default to MAX for ratio if not MIN
                    ans["status"] = "BREACH" if actual > (cov_def.limit_value or float('inf')) else "COMPLIANT"
                    
                if ans["status"] == "BREACH":
                    # Cumulative chronological search for ratio
                    tid_to_date = {}
                    for tx in matched_txns:
                        tid_to_date[tx['txn_id']] = tx['date']
                    if 'den_matched_txns' in locals():
                        for tx in den_matched_txns:
                            tid_to_date[tx['txn_id']] = tx['date']
                            
                    events = []
                    for tid, amt in converted_txns_amt:
                        events.append({'date': tid_to_date.get(tid, '2000-01-01'), 'tid': tid, 'type': 'num', 'amt': amt})
                    for tid, amt in den_converted_amt:
                        events.append({'date': tid_to_date.get(tid, '2000-01-01'), 'tid': tid, 'type': 'den', 'amt': amt})
                        
                    events.sort(key=lambda x: x['date'])
                    
                    running_num_sum = 0.0
                    running_den_sum = 0.0
                    
                    for ev in events:
                        if ev['type'] == 'num':
                            running_num_sum += ev['amt']
                        else:
                            running_den_sum += ev['amt']
                            
                        curr_den = abs(running_den_sum) if abs(running_den_sum) > 0 else 1.0
                        current_actual = round(abs(running_num_sum) / curr_den, 4)
                        
                        if cov_def.limit_type == 'MIN' and current_actual < (cov_def.limit_value or 0.0):
                            ans["evidence_txn_id"] = ev['tid']
                            break
                        elif cov_def.limit_type != 'MIN' and current_actual > (cov_def.limit_value or float('inf')):
                            ans["evidence_txn_id"] = ev['tid']
                            break

            else:
                # Aggregate for non-ratio (MAX or MIN sum)
                raw_sum = sum([amt for _, amt in converted_txns_amt])
                total = abs(raw_sum)
                ans["actual"] = round(total, 2)
                ans["evidence_txn_id"] = None
                
                if cov_def.limit_type == 'MAX' and total > (cov_def.limit_value or float('inf')):
                    ans["status"] = "BREACH"
                    # Chronological Cumulative search for MAX
                    tid_to_date = {tx['txn_id']: tx['date'] for tx in matched_txns}
                    chronological_txns = sorted(converted_txns_amt, key=lambda x: tid_to_date.get(x[0], '2000-01-01'))
                    running_sum = 0.0
                    for tid, amt in chronological_txns:
                        running_sum += amt
                        if abs(running_sum) > cov_def.limit_value:
                            ans["evidence_txn_id"] = tid
                            break
                elif cov_def.limit_type == 'MIN' and total < (cov_def.limit_value or 0.0):
                    ans["status"] = "BREACH"
                    # Chronological Cumulative search for MIN
                    tid_to_date = {tx['txn_id']: tx['date'] for tx in matched_txns}
                    chronological_txns = sorted(converted_txns_amt, key=lambda x: tid_to_date.get(x[0], '2000-01-01'))
                    running_sum = 0.0
                    for tid, amt in chronological_txns:
                        running_sum += amt
                        # For MIN limit, usually it's "must generate at least X revenue". 
                        # If total is < MIN, then there is no single transaction that "breached" it, because they all FAILED to reach the min.
                        # Wait, what if MIN is about expenses dropping below something? 
                        # In the ground truth, maybe evidence for MIN breach is None, or it's the last transaction in the period.
                        # Let's set it to None for now, as MIN breaches usually don't have a single "culprit" transaction unless it's a balance.
                        ans["evidence_txn_id"] = None
                        break
                else:
                    ans["status"] = "COMPLIANT"
                
        logger.info(f"    Result: {ans['status']} | Actual: {ans['actual']} | Evidence: {ans['evidence_txn_id']}")
        answers[cov_id] = ans
        
        # Save debug info
        debug_data["covenants"][cov_id] = {
            "definition": {
                "description": cov_def.description if cov_def else None,
                "limit_value": cov_def.limit_value if cov_def else None,
                "limit_type": cov_def.limit_type if cov_def else None,
                "transaction_category": cov_def.transaction_category if cov_def else None,
                "transaction_category_denominator": cov_def.transaction_category_denominator if cov_def else None,
                "is_single_transaction": cov_def.is_single_transaction if cov_def else None
            } if cov_def else None,
            "matched_transactions": matched_txns if 'matched_txns' in locals() else [],
            "denominator_matched_transactions": den_matched_txns if 'den_matched_txns' in locals() else [],
            "final_answer": ans
        }
        
    debug_dir = Path("debug_runs")
    debug_dir.mkdir(exist_ok=True)
    with open(debug_dir / f"{scenario_id}_debug.json", "w", encoding="utf-8") as f:
        json.dump(debug_data, f, ensure_ascii=False, indent=2)
        
    return answers

def main():
    # Ensure directories
    base_dir = Path(__file__).parent.parent
    dataset_dir = Path(os.getenv("DATASET_DIR", str(base_dir / "agentic-bank-public")))
    raw_dir = dataset_dir / "documents"
    ledger_path = dataset_dir / "master_ledger_2025.csv"
    template_path = dataset_dir / "submission_template.json"
    
    if not all(p.exists() for p in [ledger_path, template_path, raw_dir]):
        logger.error("Required files missing")
        return
        
    mapping_path = base_dir / "data/scenario_mapping.json"
    output_path = base_dir / "submission.json"
    gt_path = dataset_dir / "ground_truth.json"
    
    if not mapping_path.exists():
        logger.error("Run map_documents.py first!")
        return
        
    with open(mapping_path, "r", encoding="utf-8") as f:
        scenario_mapping = json.load(f)
        
    with open(template_path, "r", encoding="utf-8") as f:
        template = json.load(f)
        
    ledger_df = pd.read_csv(ledger_path)
    ledger_df['amount'] = pd.to_numeric(ledger_df['amount'], errors='coerce').fillna(0.0)
    
    # We need the reverse mapping from scenario -> account to get the account_id, 
    # but we can also just get it from the ledger.
    def get_account_for_scenario(scenario_id):
        row = ledger_df[ledger_df['txn_id'].str.startswith(f"TXN-{scenario_id}-")].iloc[0]
        return row['account_id']
    
    llm = LLMFactory.get_llm()
    
    answers = {}
    scenario_keys = list(template["answers"].keys())
    
    # FOR TESTING: Default to all if not specified
    scenario_filter = os.environ.get("SCENARIO_FILTER", "")
    if scenario_filter and scenario_filter.upper() != "ALL":
        filters = [f.strip() for f in scenario_filter.split(",")]
        scenario_keys = [k for k in scenario_keys if k in filters]
        logger.info(f"Filtering to run only scenario: {scenario_filter}")
        
    total_scenarios = len(scenario_keys)
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    max_workers = int(os.environ.get("MAX_WORKERS", 12))
    logger.info(f"Starting execution with {max_workers} parallel workers")
    
    def run_scenario_wrapper(scenario_id, covs):
        current_idx = scenario_keys.index(scenario_id) + 1
        
        # Resume functionality: skip if already processed
        debug_path = Path("debug_runs") / f"{scenario_id}_debug.json"
        if debug_path.exists():
            try:
                import json
                with open(debug_path, "r", encoding="utf-8") as f:
                    debug_data = json.load(f)
                    if "final_answer" in debug_data:
                        logger.info(f"Scenario {scenario_id} already processed. Resuming from cache ({current_idx}/{total_scenarios}).")
                        return scenario_id, debug_data["final_answer"]
            except Exception:
                pass # If corrupted, just re-run it
                
        doc_ids = scenario_mapping.get(scenario_id, [])
        try:
            account_id = get_account_for_scenario(scenario_id)
        except Exception:
            logger.error(f"Could not find account for {scenario_id}")
            return scenario_id, None
            
        ans = process_scenario(scenario_id, account_id, doc_ids, ledger_df, covs, raw_dir, llm, current_idx, total_scenarios)
        return scenario_id, ans

    executor = ThreadPoolExecutor(max_workers=max_workers)
    future_to_sid = {
        executor.submit(run_scenario_wrapper, sid, template["answers"][sid]): sid
        for sid in scenario_keys
    }
    
    try:
        for future in as_completed(future_to_sid):
            sid = future_to_sid[future]
            try:
                ans = future.result()
                if ans[1]:
                    answers[ans[0]] = ans[1]
            except Exception as e:
                logger.error(f"Scenario {sid} failed catastrophically: {e}")
    except KeyboardInterrupt:
        logger.warning("Received Ctrl+C (KeyboardInterrupt)! Forcefully shutting down...")
        for future in future_to_sid:
            future.cancel()
        executor.shutdown(wait=False)
        sys.exit(1)
    finally:
        executor.shutdown(wait=True)
        
    # Save submission
    template["answers"] = answers
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2)
        
    logger.info(f"Saved submission to {output_path}")
    
    if gt_path.exists():
        logger.info("Running evaluation...")
        from halyk_agent.eval.harness import run_evaluation
        run_evaluation(str(output_path), str(gt_path))
    else:
        logger.info(f"Evaluation skipped. No ground truth file found at {gt_path}.")
        
    print("\n===================================================")

if __name__ == "__main__":
    main()
