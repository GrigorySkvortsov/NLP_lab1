# Импорт библиотек
import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from nltk.tokenize import word_tokenize
from razdel import tokenize as razdel_tokenize
from nltk.stem import SnowballStemmer, PorterStemmer
import nltk
import spacy
import subprocess
import pdfkit
import os
from io import StringIO

# Загрузка ресурсов
def ensure_nltk_resources():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

# загрузка spaCy моделей
def load_spacy_model(lang):
    try:
        if lang == "Русский":
            return spacy.load("ru_core_news_sm")
        else:
            return spacy.load("en_core_web_sm")
    except OSError:
        st.warning(f"⚠️ Модель spaCy для языка {lang} не найдена. Установите её вручную.")
        return None

# Чтение корпуса
def process_corpus(input_file="3_news_corpus_universal.jsonl"): 
    texts = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                article = json.loads(line.strip())
                text = article.get("preprocessed_text", article.get("cleaned_text", article.get("text", "")))
                if text:
                    texts.append(text)
            except:
                continue
    return texts

# Токенизация и нормализация
def naive_tokenize(text):
    return text.split()

def regex_tokenize(text):
    import re
    return re.findall(r"\b\w+\b", text)

def nltk_tokenize(text, lang):
    lang = "russian" if lang == "Русский" else "english"
    return word_tokenize(text, language=lang)

def razdel_tok(text):
    return [t.text for t in razdel_tokenize(text)]

def spacy_tokenize(text, nlp):
    if nlp is None:
        return []
    doc = nlp(text)
    return [t.text for t in doc if t.text.strip()]

def porter_stem(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(t) for t in tokens]

def snowball_stem(tokens, lang):
    lang = "russian" if lang == "Русский" else "english"
    stemmer = SnowballStemmer(lang)
    return [stemmer.stem(t) for t in tokens]

def pymorphy_lemmatize(tokens):
    """Инициализация pymorphy2 с фиксом для Python 3.13"""
    import inspect
    if not hasattr(inspect, "getargspec"):
        from collections import namedtuple
        ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")
        def getargspec(func):
            spec = inspect.getfullargspec(func)
            return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)
        inspect.getargspec = getargspec

    try:
        from pymorphy2 import MorphAnalyzer
        morph = MorphAnalyzer()
        print("pymorphy2 успешно инициализирован")
        return [morph.parse(t)[0].normal_form for t in tokens]
    except Exception as e:
        print("Ошибка pymorphy2:", e)
        return None

def spacy_lemmatize(tokens, nlp):
    if nlp is None:
        return tokens
    doc = nlp(" ".join(tokens))
    return [t.lemma_ for t in doc]

# Метрики и отчёты
def compute_metrics(tokens_list, vocab):
    token_lengths = [len(t) for toks in tokens_list for t in toks]
    total_tokens = len(token_lengths)
    oov_tokens = sum(1 for toks in tokens_list for t in toks if t not in vocab)
    oov_percentage = oov_tokens / total_tokens * 100 if total_tokens > 0 else 0

    freq = Counter(t for toks in tokens_list for t in toks)
    top_tokens = dict(sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10])

    return {
        "token_lengths": token_lengths,
        "oov_percentage": oov_percentage,
        "token_freq": top_tokens,
        "vocab_size": len(vocab)
    }

def generate_report(metrics, tokenizer, normalizer, lang):
    report = f"# Отчёт по обработке текста\n\n"
    report += f"**Язык:** {lang}\n"
    report += f"**Токенизация:** {tokenizer}\n"
    report += f"**Нормализация:** {normalizer}\n\n"
    report += f"**Размер словаря:** {metrics['vocab_size']}\n"
    report += f"**Доля OOV:** {metrics['oov_percentage']:.2f}%\n\n"
    report += "## Частотность токенов (топ-10)\n"
    report += "| Токен | Частота |\n|--------|---------|\n"
    for token, freq in metrics["token_freq"].items():
        report += f"| {token} | {freq} |\n"
    return report

# Основное приложение Streamlit
def main():
    st.title("🧠 Интерактивный анализ токенизации и нормализации текста")

    ensure_nltk_resources()

    # выбор языка
    lang = st.selectbox("Выберите язык текста", ["Русский", "Английский"])

    # загрузка корпуса
    st.subheader("📁 Загрузка корпуса")
    use_default = st.checkbox("Использовать предзагруженный корпус")
    uploaded = st.file_uploader("Или загрузите свой .jsonl файл", type=["jsonl"])

    if use_default:
        corpus_path = "3_news_corpus_universal.jsonl"
    elif uploaded:
        corpus_path = "uploaded_corpus.jsonl"
        with open(corpus_path, "wb") as f:
            f.write(uploaded.getbuffer())
    else:
        st.stop()

    texts = process_corpus(corpus_path)
    if not texts:
        st.error("Не удалось прочитать корпус.")
        st.stop()
    st.success(f"Загружено {len(texts)} текстов")

    # загрузка модели spaCy
    nlp = load_spacy_model(lang)

    # выбор методов
    st.subheader("⚙️ Настройки обработки")
    tokenizer = st.selectbox("Метод токенизации", ["Наивная", "Регулярные выражения", "NLTK", "razdel", "spaCy"])
    normalizer = st.selectbox("Метод нормализации", ["Без нормализации", "PorterStemmer", "SnowballStemmer", "pymorphy2", "spaCy Lemmatizer"])

    # обработка корпуса
    if st.button("🔍 Выполнить обработку"):
        tokens_list = []
        vocab = set()

        progress = st.progress(0)
        for i, text in enumerate(texts):
            # токенизация
            if tokenizer == "Наивная":
                tokens = naive_tokenize(text)
            elif tokenizer == "Регулярные выражения":
                tokens = regex_tokenize(text)
            elif tokenizer == "NLTK":
                tokens = nltk_tokenize(text, lang)
            elif tokenizer == "razdel":
                tokens = razdel_tok(text)
            elif tokenizer == "spaCy":
                tokens = spacy_tokenize(text, nlp)
            else:
                tokens = []

            # нормализация
            if normalizer == "PorterStemmer":
                tokens = porter_stem(tokens)
            elif normalizer == "SnowballStemmer":
                tokens = snowball_stem(tokens, lang)
            elif normalizer == "pymorphy2":
                tokens = pymorphy_lemmatize(tokens)
            elif normalizer == "spaCy Lemmatizer":
                tokens = spacy_lemmatize(tokens, nlp)

            if tokens:
                tokens_list.append(tokens)
                vocab.update(tokens)
            progress.progress((i + 1) / len(texts))

        # вычисление метрик
        metrics = compute_metrics(tokens_list, vocab)

        # визуализация
        st.subheader("📊 Результаты и визуализация")

        fig1 = px.histogram(metrics["token_lengths"], nbins=50, title="Распределение длин токенов")
        st.plotly_chart(fig1)
        fig1.write_image("token_lengths.png")

        freq_df = pd.DataFrame(metrics["token_freq"].items(), columns=["Токен", "Частота"])
        fig2 = px.bar(freq_df, x="Токен", y="Частота", title="Частотность токенов (топ-10)")
        st.plotly_chart(fig2)

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name="OOV", x=["OOV"], y=[metrics["oov_percentage"]]))
        fig3.add_trace(go.Bar(name="In-Vocab", x=["In-Vocab"], y=[100 - metrics["oov_percentage"]]))
        fig3.update_layout(title="Доля OOV", barmode="stack")
        st.plotly_chart(fig3)

        # отчёт
        report = generate_report(metrics, tokenizer, normalizer, lang)
        st.markdown(report)

        # экспорт
        st.subheader("📤 Экспорт отчёта")
        with open("report.html", "w", encoding="utf-8") as f:
            f.write(f"<html><body>{report}</body></html>")

        with open("report.html", "r", encoding="utf-8") as f:
            st.download_button("⬇️ Скачать HTML", f, file_name="report.html")

        with open("report.html", "r", encoding="utf-8") as f:
            st.download_button("⬇️ Скачать pdf", f, file_name="report.html")

        # try:
        #     path_wkhtmltopdf = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
        #     config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
        #     pdfkit.from_file("report.html", "report.pdf", configuration=config)
        #     with open("report.pdf", "rb") as f:
        #         st.download_button("⬇️ Скачать PDF", f, file_name="report.pdf", mime="application/pdf")
        # except Exception as e:
        #     st.warning(f"Не удалось создать PDF: {str(e)}")


# =====================[ Запуск ]=====================
if __name__ == "__main__":
    main()
