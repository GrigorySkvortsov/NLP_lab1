#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
from pathlib import Path
from tqdm import tqdm

try:
    from razdel import tokenize as razdel_tokenize
except Exception as e:
    raise RuntimeError("Установи razdel: pip install razdel") from e

try:
    import spacy
    nlp_ru = spacy.load("ru_core_news_sm")
except Exception as e:
    raise RuntimeError("Установи spaCy и модель ru_core_news_sm: python -m spacy download ru_core_news_sm") from e

IN_PATH = Path("3_news_corpus_universal.jsonl")
OUT_PATH = Path("outputs/processed_razdel_spacy_final.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

_re_url = re.compile(r"""(?i)\b((?:https?://|www\.)[^\s<>"'()]+)""", flags=re.UNICODE)
_re_email = re.compile(r"(?i)\b[\w\.-]+@[\w\.-]+\.\w+\b")
_re_digits = re.compile(r"(?<!<)\b\d[\d\.,\-]*\b(?!>)")

def normalize_placeholders(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"<\s*NUM\s*>", "<NUM>", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*URL\s*>", "<URL>", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*EMAIL\s*>", "<EMAIL>", text, flags=re.IGNORECASE)
    text = _re_url.sub("<URL>", text)
    text = _re_email.sub("<EMAIL>", text)
    text = _re_digits.sub("<NUM>", text)
    return text

def razdel_tokenize_with_punct(text):
    return [t.text for t in razdel_tokenize(text)]

def merge_angle_placeholders(tokens):
    out = []
    i = 0
    L = len(tokens)
    while i < L:
        if tokens[i] == "<":
            j = i + 1
            collected = []
            while j < L and tokens[j] != ">":
                collected.append(tokens[j])
                j += 1
            if j < L and collected:
                core = "".join(collected).strip().upper()
                core = re.sub(r"[^\w]", "", core)
                if core in {"NUM", "URL", "EMAIL"}:
                    out.append(f"<{core}>")
                    i = j + 1
                    continue
            out.append(tokens[i])
            i += 1
        else:
            out.append(tokens[i])
            i += 1
    return out

# ---- НОВАЯ функция: лемматизация ПО токенам ----
def lemmatize_from_tokens(tokens):
    """
    Для каждого токена:
     - если это placeholder (<NUM>, <URL>, <EMAIL>) — вернуть как есть;
     - если это пунктуация — пропустить;
     - иначе — получить лемму через spaCy (обрабатываем token как отдельный текст).
    Это гарантирует, что placeholders не будут распилены и не превратятся в '< num >'.
    """
    lemmas = []
    for tok in tokens:
        if not tok:
            continue
        # если уже placeholder — добавляем как есть
        if tok in {"<NUM>", "<URL>", "<EMAIL>"}:
            lemmas.append(tok)
            continue
        # если токен — одиночный знак пунктуации или полностью пунктуационный — пропускаем
        if all(ch in '.,:;!?()[]{}«»„“"\'`—–-<>/\\@#%&' for ch in tok):
            continue
        # теперь получаем лемму токена: используем nlp_ru на самом токене
        # безопасно: nlp_ru(token) обычно даёт один токен в doc
        doc = nlp_ru(tok)
        # находясь в doc, найдём первый невызовный токен (м.б. пробелы/пунктуация)
        found = False
        for t in doc:
            if t.is_space:
                continue
            if t.like_url:
                lemmas.append("<URL>")
                found = True
                break
            if t.like_email:
                lemmas.append("<EMAIL>")
                found = True
                break
            if t.like_num:
                lemmas.append("<NUM>")
                found = True
                break
            if t.is_punct:
                # пропустить
                found = True
                break
            lemma = t.lemma_.strip().lower()
            if lemma:
                lemmas.append(lemma)
                found = True
                break
        if not found:
            # fallback — добавим оригинал в lower
            lemmas.append(tok.lower())
    return lemmas

# --- Основной цикл (обновлённая последовательность) ---
if not IN_PATH.exists():
    raise FileNotFoundError(f"Файл {IN_PATH} не найден!")

with IN_PATH.open("r", encoding="utf-8") as fin, OUT_PATH.open("w", encoding="utf-8") as fout:
    for line in tqdm(fin, desc="processing"):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue

        for field in ("title", "text"):
            original = rec.get(field, "") or ""
            # 1) нормализуем placeholders в тексте
            norm = normalize_placeholders(original)
            # 2) токенизируем razdel (с пунктуацией)
            tokens = razdel_tokenize_with_punct(norm)
            # 3) собираем разбитые < NUM > обратно
            tokens = merge_angle_placeholders(tokens)
            # 4) лемматизируем по токенам (гарантируем целостность placeholders)
            lemmas = lemmatize_from_tokens(tokens)
            # записываем результат обратно в поле
            rec[field] = " ".join(lemmas)

        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

print("Готово. Сохранено в", OUT_PATH)
