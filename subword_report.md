# Отчёт по сравнительному анализу подсловных токенизаторов

Корпус: 3_news_corpus_universal.jsonl
Модели: BPE, WordPiece, Unigram. Даты запуска эксперимента: Fri Oct 24 20:37:16 2025

| model | model_type | vocab_size | vocab_size_actual | fragmentation_% | oov_% | avg_subtokens_per_word | compression_ratio | reconstruction_cosine_% | reconstruction_exact_% | training_time_s | time_per_1000_articles_s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bpe_vocab8000 | bpe | 8000 | 8000 | 55.80 | 0.00 | 1.906 | 1.906 | 80.08 | 0.00 | 0.57 | 0.57 |
| bpe_vocab16000 | bpe | 16000 | 16000 | 42.95 | 0.00 | 1.632 | 1.632 | 90.61 | 0.00 | 0.91 | 0.91 |
| bpe_vocab32000 | bpe | 32000 | 32000 | 35.22 | 0.00 | 1.495 | 1.495 | 94.70 | 0.00 | 1.02 | 1.02 |
| wordpiece_vocab8000 | wordpiece | 8000 | 8000 | 57.20 | 0.00 | 1.981 | 1.981 | 70.44 | 0.00 | 0.77 | 0.77 |
| wordpiece_vocab16000 | wordpiece | 16000 | 16000 | 44.41 | 0.00 | 1.687 | 1.687 | 82.13 | 0.00 | 0.94 | 0.94 |
| wordpiece_vocab32000 | wordpiece | 32000 | 32000 | 35.44 | 0.00 | 1.518 | 1.518 | 89.46 | 0.00 | 1.08 | 1.08 |
| unigram_vocab8000 | unigram | 8000 | 8000 | 68.37 | 0.00 | 2.135 | 2.135 | 81.98 | 0.00 | 3.31 | 3.31 |
| unigram_vocab16000 | unigram | 16000 | 16000 | 65.36 | 0.00 | 1.977 | 1.977 | 85.11 | 0.00 | 2.19 | 2.19 |
| unigram_vocab32000 | unigram | 32000 | 32000 | 65.15 | 0.00 | 1.965 | 1.965 | 85.29 | 0.00 | 1.10 | 1.10 |
