# Montessori Early Childhood Strict RAG Assistant

Локальная Strict RAG-система для ответов на вопросы по методике Монтессори (0–3 года) строго по первоисточникам, без галлюцинаций и использования внешних знаний.

## Архитектура
- **LLM:** `unsloth/Qwen2.5-3B-Instruct` (4-bit quantization).
- **Vector DB:** ChromaDB.
- **Embeddings:** `intfloat/multilingual-e5-small`.
- **Защита от галлюцинаций:** Порог сходства ретривера (score_threshold = 0.5) + детерминированная генерация (`temperature=0.01`).

## Запуск
```bash
pip install -r requirements.txt
python main.py
