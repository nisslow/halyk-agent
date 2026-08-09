import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

# Ensure imports work from project root
sys.path.append(os.path.dirname(__file__))

from agent.orchestrator import graph

def run_integration_test():
    print("==================================================")
    print("🚀 Запуск интеграционного теста LangGraph Оркестратора")
    print("==================================================\n")
    
    # Test Case 1: Valid transaction under Old Tariff (1.5%)
    # Amount 1000, fee is 15
    print(">>> ТЕСТ 1: Транзакция в январе 2024 (Сумма: 1000, Комиссия: 15)")
    initial_state_1 = {
        "query": "Проверь транзакцию",
        "transaction_id": "TXN001",
        "transaction_date": "2024-01-15",
        "transaction_amount": 1000.0,
        "transaction_fee_applied": 15.0,
        "retries": 0
    }
    
    result_1 = graph.invoke(initial_state_1)
    print(f"\n[ФИНАЛЬНЫЙ ОТВЕТ АГЕНТА]: {result_1['final_answer']}\n")
    
    # Test Case 2: Invalid transaction under New Tariff (1.0%)
    # Amount 1000, fee is 15 (but should be max 10.0 according to 1.0% rule!)
    print(">>> ТЕСТ 2: Транзакция в июне 2024 (Сумма: 1000, Комиссия: 15)")
    initial_state_2 = {
        "query": "Проверь транзакцию",
        "transaction_id": "TXN003",
        "transaction_date": "2024-06-20",
        "transaction_amount": 1000.0,
        "transaction_fee_applied": 15.0, # This is a violation! 1000 * 1% = 10
        "retries": 0
    }
    
    result_2 = graph.invoke(initial_state_2)
    print(f"\n[ФИНАЛЬНЫЙ ОТВЕТ АГЕНТА]: {result_2['final_answer']}\n")

if __name__ == "__main__":
    run_integration_test()
