import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_openai import ChatOpenAI

class TariffRule(BaseModel):
    operation_type: str = Field(description="The type of operation (e.g., Transfer, Withdrawal)")
    currency: str = Field(description="Currency of the operation, e.g. KZT, USD")
    fee_percent: float = Field(description="The commission percentage")

class DocumentMetadata(BaseModel):
    version: str = Field(description="The version of the document")
    rules: List[TariffRule] = Field(description="List of extracted tariff rules")

class ExtractionResult(BaseModel):
    documents: List[DocumentMetadata]

MODELS = [
    "inclusionai/ling-3.0-flash:free",
    "poolside/laguna-s-2.1:free",
    "nvidia/nemotron-3-super-120b-a12b:free"
]

markdown_content = """
# Тарифы Halyk Bank для юридических лиц
**Версия:** 2.0 (Актуальная)
**Действует с:** 2024-06-01

## 2. Таблицы комиссий
| Тип операции | Валюта | Комиссия (%) |
|--------------|--------|--------------|
| Перевод      | KZT    | 1.0%         |
| Снятие       | KZT    | 3.0%         |
"""

prompt = f"Extract tariff rules from this document:\n{markdown_content}"

print("=== Тестирование моделей OpenRouter на поддержку Structured Output ===")

for model_name in MODELS:
    print(f"\nТестируем модель: {model_name}")
    try:
        llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY", "your_key_here"),
            model=model_name,
            temperature=0
        )
        
        structured_llm = llm.with_structured_output(ExtractionResult)
        result = structured_llm.invoke(prompt)
        
        print("✅ УСПЕХ! Модель успешно вернула типизированный JSON:")
        for doc in result.documents:
            print(f"Версия: {doc.version}")
            for r in doc.rules:
                print(f"  - {r.operation_type}: {r.fee_percent}%")
                
    except Exception as e:
        print(f"❌ ОШИБКА ({type(e).__name__}): {e}")

