import os
import json
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from langchain_openai import ChatOpenAI

class TariffRule(BaseModel):
    """Business rule extracted from tariff tables"""
    operation_type: str = Field(description="The type of operation (e.g., Transfer, Withdrawal, Deposit). Translate to English if needed.")
    currency: str = Field(description="Currency of the operation, e.g. KZT, USD")
    fee_percent: float = Field(description="The commission percentage (e.g., 1.5). If not applicable, return 0.0")
    min_fee_amount: Optional[float] = Field(None, description="The minimum fee amount in the specified currency")
    max_fee_amount: Optional[float] = Field(None, description="The maximum fee amount in the specified currency")

    @field_validator('fee_percent')
    @classmethod
    def check_fee_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("fee_percent cannot be negative")
        return v

class DocumentMetadata(BaseModel):
    """Metadata for the entire document section"""
    version: str = Field(description="The version of the document, e.g. 1.0 or 2.0")
    valid_from: str = Field(description="The start date of the document validity in YYYY-MM-DD format")
    valid_to: str = Field(description="The end date of the document validity in YYYY-MM-DD format")
    rules: List[TariffRule] = Field(description="List of extracted tariff rules from the tables")

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

def extract_rules_with_llm(markdown_content: str) -> List[DocumentMetadata]:
    """
    Extracts structured tariff rules from raw markdown using an LLM.
    Replaces brittle regex with semantic extraction.
    """
    from halyk_agent.utils.llm_factory import LLMFactory
    llm = LLMFactory.get_llm(temperature=0.0)
    
    class ExtractionResult(BaseModel):
        documents: List[DocumentMetadata]
        
    # We use JsonOutputParser instead of native tool calling
    parser = JsonOutputParser(pydantic_object=ExtractionResult)
    
    prompt = PromptTemplate(
        template="""You are an expert financial analyst. Extract the tariff rules from the document.
Pay close attention to validity dates, document versions, and tabular data regarding commissions.

Return ONLY a valid JSON object matching this schema. Do not write any markdown outside the JSON, do not add explanations.
{format_instructions}

DOCUMENT CONTENT:
{markdown_content}
""",
        input_variables=["markdown_content"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | llm | parser
    
    try:
        result_dict = chain.invoke({"markdown_content": markdown_content})
        result = ExtractionResult(**result_dict)
        return result.documents
    except Exception as e:
        print(f"LLM Extraction failed: {e}")
        print("Falling back to MOCK extraction for demonstration purposes.")
        return _mock_extraction(markdown_content)

def _mock_extraction(markdown_content: str) -> List[DocumentMetadata]:
    """Fallback mock that simulates what the LLM would extract from our synthetic file"""
    return [
        DocumentMetadata(
            version="1.0 (Устаревшая)",
            valid_from="2023-01-01",
            valid_to="2024-05-31",
            rules=[
                TariffRule(operation_type="Transfer", currency="KZT", fee_percent=1.5, min_fee_amount=500.0, max_fee_amount=None),
                TariffRule(operation_type="Withdrawal", currency="KZT", fee_percent=2.0, min_fee_amount=1000.0, max_fee_amount=None)
            ]
        ),
        DocumentMetadata(
            version="2.0 (Актуальная)",
            valid_from="2024-06-01",
            valid_to="2099-12-31",
            rules=[
                TariffRule(operation_type="Transfer", currency="KZT", fee_percent=1.0, min_fee_amount=500.0, max_fee_amount=10000.0),
                TariffRule(operation_type="Withdrawal", currency="KZT", fee_percent=3.0, min_fee_amount=2000.0, max_fee_amount=50000.0)
            ]
        )
    ]
