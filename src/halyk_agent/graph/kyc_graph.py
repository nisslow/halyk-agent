import kuzu
from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from loguru import logger
import os
import hashlib
import json

class AffiliateList(BaseModel):
    borrower_name: str = Field(description="The exact name of the Borrower company")
    affiliates: List[str] = Field(description="List of exact names of affiliated companies, subsidiaries, or related parties")

def init_kuzu_db(db_path: str = "kuzu_kyc_db"):
    db = kuzu.Database(db_path)
    conn = kuzu.Connection(db)
    
    try:
        conn.execute("CREATE NODE TABLE Company (name STRING, PRIMARY KEY (name))")
        conn.execute("CREATE REL TABLE IS_AFFILIATE_OF (FROM Company TO Company)")
    except RuntimeError as e:
        if "already exists" not in str(e).lower():
            logger.warning(f"KuzuDB init warning: {e}")
            
    return conn

KYC_CACHE_PATH = "data/kyc_cache.json"

def extract_and_store_affiliates(llm, doc_text: str, conn: kuzu.Connection):
    if os.path.exists(KYC_CACHE_PATH):
        with open(KYC_CACHE_PATH, "r", encoding="utf-8") as f:
            KYC_CACHE = json.load(f)
    else:
        KYC_CACHE = {}

    doc_hash = hashlib.md5(doc_text.encode("utf-8")).hexdigest()
    
    if doc_hash in KYC_CACHE:
        aff_data = KYC_CACHE[doc_hash]
        logger.info("Using cached affiliates")
    else:
        parser = JsonOutputParser(pydantic_object=AffiliateList)
        prompt = PromptTemplate(
            template='''You are a compliance officer. Read the KYC/Compliance dossier below and extract the Borrower and all its Affiliated/Related Parties.
            
CRITICAL RULE: If the document specifies a threshold for recognizing a related party (e.g. "owns 25.0% or more", or "владеет 25.0% и более"), you MUST ONLY extract entities that meet or exceed this exact threshold. Do NOT include entities that are below the ownership threshold.

KYC Document:
{doc_text}

Return JSON strictly matching:
{format_instructions}''',
            input_variables=["doc_text"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        try:
            res = llm.invoke(prompt.format(doc_text=doc_text[:15000]))
            aff_data = parser.parse(res.content)
            
            KYC_CACHE[doc_hash] = aff_data
            os.makedirs("data", exist_ok=True)
            with open(KYC_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(KYC_CACHE, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed LLM parsing for KYC: {e}")
            return
            
    try:
        borrower = aff_data["borrower_name"]
        conn.execute("MERGE (c:Company {name: $name})", {"name": borrower})
        for aff in aff_data["affiliates"]:
            conn.execute("MERGE (c:Company {name: $name})", {"name": aff})
            conn.execute(
                "MATCH (a:Company {name: $aff}), (b:Company {name: $bor}) "
                "MERGE (a)-[:IS_AFFILIATE_OF]->(b)",
                {"aff": aff, "bor": borrower}
            )
        logger.info(f"Stored {len(aff_data['affiliates'])} affiliates for {borrower} in KuzuDB")
    except Exception as e:
        logger.error(f"Failed to extract/store affiliates: {e}")

def get_affiliates_for_scenario(conn: kuzu.Connection) -> List[str]:
    try:
        res = conn.execute("MATCH (a:Company)-[:IS_AFFILIATE_OF]->(b:Company) RETURN DISTINCT a.name")
        affiliates = []
        while res.has_next():
            affiliates.append(res.get_next()[0])
        return affiliates
    except Exception as e:
        logger.error(f"Failed to query KuzuDB: {e}")
        return []
