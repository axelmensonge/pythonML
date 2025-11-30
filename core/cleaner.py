import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
nlp = spacy.load("fr_core_news_sm")

stop_fr = set(stopwords.words("french"))
stop_en = set(stopwords.words("english"))

def clean_basic(text):
    if not text:
        return ""
    return text.lower().strip()


def normalize_text(text):
    if not text:
        return ""

    text = clean_basic(text)
    doc = nlp(text)

    tokens = []
    for tok in doc:
        if any(c.isdigit() for c in tok.text) or "-" in tok.text.upper():
            tokens.append(tok.text.lower())
            continue

        if tok.is_alpha and tok.text not in stop_fr and tok.text not in stop_en:
            tokens.append(tok.lemma_)

    return " ".join(tokens)


def clean_category(cat):
    if isinstance(cat, list):
        cat = " ".join(cat)
    if not cat:
        return ""
    return clean_basic(cat)


def preprocess_dataframe(dfs):
    df = pd.concat(dfs.values(), ignore_index=True)

    df = df[df["id"].notna()]

    df["title_clean"] = df["title"].apply(clean_basic)

    df["text_raw_clean"] = df["text"].apply(clean_basic)

    df["text_merged"] = df.apply(
        lambda row: row["text_raw_clean"] if row["text_raw_clean"] else row["title_clean"],
        axis=1
    )

    df["text_clean"] = df["text_merged"].apply(normalize_text)

    df["category_clean"] = df["category"].apply(clean_category)

    df = df[df["text_clean"].str.strip() != ""]

    return df