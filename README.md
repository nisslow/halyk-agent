# Halyk AI Challenge 2026 — Autonomous Covenant Compliance Agent

> AI-агент для автоматической проверки соблюдения кредитных ковенантов на основе PDF-договоров и транзакционных данных.

## 🏗️ Как это работает

Агент получает на вход набор PDF-документов (кредитные договоры, дополнения, KYC, аудиторские заключения) и реестр транзакций (`master_ledger_2025.csv`). Для каждого заёмщика он:

1. **Маппит документы на сценарии** — сканирует PDF через PyMuPDF, находит `ACC-XXXX` и привязывает документы к заёмщикам.
2. **Извлекает метаданные** — определяет тип документа (CONTRACT / AMENDMENT / KYC / AUDIT), даты действия, и отфильтровывает устаревшие версии.
3. **Парсит ковенанты через LLM** — отправляет релевантные фрагменты договора в нейросеть, которая возвращает структурированное определение: описание, лимит, тип (MAX / MIN / RATIO), категории транзакций.
4. **Классифицирует транзакции** — двухуровневая система: сначала эвристический фильтр по ключевым словам, затем LLM-батч-классификация для точного отбора.
5. **Вычисляет метрики** — суммирует подходящие транзакции, применяет обменные курсы, вычисляет коэффициенты (RATIO).
6. **Выносит вердикт** — сравнивает фактическое значение с лимитом и определяет `COMPLIANT` или `BREACH`, указывая конкретную транзакцию-нарушителя.

```
┌────────────────────────────────────────────────────────────────────┐
│                     HALYK AGENT PIPELINE                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐       │
│  │ MAP DOCS │──▶│ PARSE    │──▶│ EXTRACT  │──▶│ CLASSIFY │       │
│  │          │   │ METADATA │   │ COVENANTS│   │   TXNS   │       │
│  │ PyMuPDF  │   │ LLM +    │   │ LLM +    │   │ Heuristic│       │
│  │ ACC-XXXX │   │ Temporal │   │ Pydantic │   │ + LLM    │       │
│  │ matching │   │ Filtering│   │ Schema   │   │ Batch    │       │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘       │
│       │                                             │              │
│       ▼                                             ▼              │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │              CALCULATE & DECIDE                         │      │
│  │  • Sum matched transactions (with currency conversion)  │      │
│  │  • Compute ratios (numerator / denominator)             │      │
│  │  • Compare actual vs limit → COMPLIANT / BREACH         │      │
│  │  • Identify evidence transaction for breaches           │      │
│  └─────────────────────────────────────────────────────────┘      │
│       │                                                            │
│       ▼                                                            │
│  ┌──────────┐                                                      │
│  │ OUTPUT   │  submission.json                                     │
│  │          │  (status, actual, evidence_txn_id per covenant)      │
│  └──────────┘                                                      │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

> **Важно:** В репозитории реализовано **два режима работы**:
>
> | Режим | Точка входа | Retrieval | Инфраструктура |
> |-------|-------------|-----------|----------------|
> | **Полный Hybrid RAG** (production) | `scripts/run_pipeline.py` / `halyk_agent.main` | bge-m3 (dense+sparse+colbert) + BM25 + Qdrant + temporal filtering | Qdrant, GPU для эмбеддингов, предварительная ингистация |
> | **Хакатон LLM-only** (быстро, без инфраструктуры) | `run_agent.bat` → `scripts/run_hackathon_agent.py` | Прямой текст PDF → контекст LLM (PyMuPDF) | Только Python + API OpenRouter/LM Studio |
>
> `run_agent.bat`, используемый для сабмишн в челлендже, запускает **LLM-only режим** для скорости и нулевой инфраструктуры. Полный Hybrid RAG пайплайн реализован в `src/halyk_agent/retrieval/hybrid_retriever.py` и оркестрируется через `scripts/run_pipeline.py`.
>
> **Тестирование:** Разработка и отладка велась локально в режиме **Хакатон LLM-only** (бесплатные/дешёвые модели через OpenRouter/LM Studio). **Финальный прогон сабмишн** выполнялся на платной модели (DeepSeek V4 Flash / Nemotron 3 Ultra) для максимального качества.

## ✨ Ключевые особенности

| Компонент | Технология | Описание |
|-----------|-----------|----------|
| **PDF-парсинг** | `PyMuPDF (fitz)` | Быстрое извлечение текста из PDF; автоматический фоллбэк на OCR при нечитаемых документах |
| **OCR** | `EasyOCR` (ru + en) | Распознавание отсканированных документов на русском и английском языках |
| **LLM** | `OpenRouter` (DeepSeek V4 Flash) | Извлечение ковенантов и классификация транзакций через облачную нейросеть |
| **Структурированный вывод** | `Pydantic` + `LangChain` | Гарантированный JSON-ответ от LLM через `with_structured_output()` |
| **Отказоустойчивость** | `tenacity` | Автоматические ретраи с экспоненциальной задержкой (до 10 попыток) |
| **Ротация ключей** | Встроенная | Поддержка нескольких API-ключей с автоматическим переключением при ошибках |
| **Кэширование** | JSON-кэш | Кэширование метаданных документов и определений ковенантов между запусками |
| **Параллелизация** | `ThreadPoolExecutor` | До 12 заёмщиков обрабатываются одновременно |
| **Возобновление** | `debug_runs/` | При перезапуске скрипт пропускает уже обработанных заёмщиков |
| **KYC / Аффилированные лица** | `KuzuDB` | Граф связанных лиц для проверки ковенантов по related party |

## 🚀 Быстрый старт

### Требования

- **Python 3.11+**
- **Windows / Linux / macOS**
- **API-ключ OpenRouter** (зарегистрируйтесь на [openrouter.ai](https://openrouter.ai))

### Установка

```bash
# 1. Клонируйте репозиторий
git clone https://github.com/nisslow/halyk-agent.git
cd halyk-agent

# 2. Создайте виртуальное окружение
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Установите зависимости
pip install -r requirements.txt

# 4. Установите сам пакет
pip install -e .
```

### Настройка

Отредактируйте файл `run_agent.bat` (Windows) и укажите:

```bat
REM Путь к папке с датасетом
set DATASET_DIR=E:\path\to\agentic-bank-hidden

REM Ваш API-ключ OpenRouter
set OPENROUTER_API_KEY=sk-or-v1-your-key-here

REM Модель (по умолчанию DeepSeek V4 Flash — лучшая по соотношению цена/качество)
set LLM_MODEL=deepseek/deepseek-v4-flash-0731

REM Количество параллельных потоков (по умолчанию 12)
set MAX_WORKERS=12
```

### Запуск

```bash
# Windows — просто двойной клик:
run_agent.bat

# Или вручную из терминала:
python scripts/map_documents.py
python scripts/run_hackathon_agent.py
```

Скрипт выполнит два этапа:
1. `map_documents.py` — сканирует все PDF и создаёт маппинг документов на заёмщиков.
2. `run_hackathon_agent.py` — запускает основной пайплайн анализа ковенантов.

Результат сохраняется в файл **`submission.json`** в корне проекта.

### Использование локальной LLM (опционально)

Если у вас есть локальная модель через LM Studio или Ollama:

```bat
set LLM_PROVIDER=local
set LLM_MODEL=your-local-model-name
set LOCAL_BASE_URL=http://localhost:1234/v1
```

## 📁 Структура проекта

```
halyk-agent/
├── scripts/
│   ├── map_documents.py              # Маппинг PDF → сценарий (по ACC-XXXX)
│   └── run_hackathon_agent.py         # Основной пайплайн (900 строк)
├── src/halyk_agent/
│   ├── eval/
│   │   └── harness.py                 # Оценка submission vs ground_truth
│   ├── graph/
│   │   ├── kyc_graph.py               # KuzuDB граф аффилированных лиц
│   │   └── entity_resolution.py       # Разрешение сущностей (RapidFuzz)
│   ├── ingestion/
│   │   ├── pdf_pipeline.py            # PDF-парсинг (PyMuPDF + OCR)
│   │   ├── transaction_loader.py      # Загрузка транзакций из CSV
│   │   └── llm_extractor.py           # LLM-экстракция из документов
│   ├── retrieval/
│   │   └── hybrid_retriever.py        # Гибридный поиск (BM25 + embeddings)
│   ├── utils/
│   │   ├── llm_factory.py             # Фабрика LLM (OpenRouter / Local)
│   │   └── ocr_engine.py              # EasyOCR движок (ru + en)
│   ├── validation/
│   │   └── z3_validator.py            # Z3 SMT валидатор бизнес-правил
│   ├── proof/
│   │   └── consistency.py             # Проверка консистентности
│   └── main.py                        # CLI точка входа
├── run_agent.bat                      # Точка запуска (Windows)
├── .env.example                       # Шаблон переменных окружения
├── requirements.txt                   # Python-зависимости
├── pyproject.toml                     # Конфигурация пакета
├── Dockerfile                         # Docker-образ
└── README.md
```

## 📋 Формат выходного файла (submission.json)

```json
{
  "team": "Название команды",
  "contact_email": "email@example.com",
  "model": "deepseek/deepseek-v4-flash-0731",
  "answers": {
    "S1": {
      "6.1": {
        "status": "COMPLIANT",
        "actual": 0.041,
        "evidence_txn_id": null
      },
      "6.2": {
        "status": "BREACH",
        "actual": 2623107.95,
        "evidence_txn_id": "TXN-S1-0054"
      }
    }
  }
}
```

Каждый ковенант содержит:
- **`status`** — `"COMPLIANT"` (соблюдён) или `"BREACH"` (нарушен)
- **`actual`** — фактическое числовое значение метрики
- **`evidence_txn_id`** — ID транзакции-нарушителя (или `null` при соблюдении)

## 🔧 Переменные окружения

| Переменная | По умолчанию | Описание |
|-----------|-------------|----------|
| `DATASET_DIR` | `./agentic-bank-public` | Путь к папке с датасетом |
| `LLM_PROVIDER` | `openrouter` | Провайдер LLM (`openrouter` или `local`) |
| `LLM_MODEL` | `deepseek/deepseek-v4-flash-0731` | Название модели |
| `OPENROUTER_API_KEY` | — | API-ключ OpenRouter |
| `MAX_WORKERS` | `12` | Количество параллельных потоков |
| `SCENARIO_FILTER` | — | Фильтр сценариев (например, `S1,S2,B3`) |
| `LOCAL_BASE_URL` | `http://localhost:1234/v1` | URL локальной модели |

## 🧪 Оценка результатов

Если в папке датасета есть файл `ground_truth.json`, скрипт автоматически запустит оценку:

```bash
# Или вручную:
python src/halyk_agent/eval/harness.py ground_truth.json submission.json
```

Система оценки (на каждый ковенант — макс 1.0 балл):
- **0.50** — правильный статус (COMPLIANT / BREACH)
- **0.30** — точность числового значения (actual) с допуском 5%
- **0.20** — правильная идентификация транзакции-нарушителя

## 📝 Лицензия

MIT License

---

**Built for Halyk AI Challenge 2026** 🏆