import os
import sys

# Ensure proper encoding for windows terminal
sys.stdout.reconfigure(encoding='utf-8')

from ingestion.llm_extractor import extract_rules_with_llm

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TARIFF_FILE = os.path.join(DATA_DIR, "synthetic_tariffs_parsed.md")

def run_test():
    print("=== Запуск теста LLM Structured Extraction ===")
    
    if not os.path.exists(TARIFF_FILE):
        print(f"Error: {TARIFF_FILE} not found.")
        return
        
    print("\n1. Чтение сырого Markdown файла...")
    with open(TARIFF_FILE, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
        
    print("2. Передача текста в LLM (или Mock) для парсинга через Pydantic...")
    documents = extract_rules_with_llm(markdown_content)
    
    print("\n3. Результат извлечения (Валидированные Python-объекты):")
    for doc in documents:
        print(f"\n--- Документ: {doc.version} ---")
        print(f"Действует: с {doc.valid_from} по {doc.valid_to}")
        print("Правила (Тарифы):")
        for rule in doc.rules:
            print(f"  - Операция: {rule.operation_type}")
            print(f"    Валюта: {rule.currency}")
            print(f"    Комиссия: {rule.fee_percent}%")
            print(f"    Мин. сумма: {rule.min_fee_amount}")
            print(f"    Макс. сумма: {rule.max_fee_amount}")
            
    print("\n=== Тест завершен успешно! ===")
    print("Обратите внимание: мы получили чистые структурированные данные без единого регулярного выражения (Regex)!")

if __name__ == "__main__":
    run_test()
