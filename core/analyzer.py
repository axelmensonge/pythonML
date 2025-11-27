import json
from pathlib import Path
from collections import Counter
from core.logger import get_logger
from core.config import SUMMARY_FILE
import pandas as p


logger = get_logger(__name__)

def compute_text_length(df: p.DataFrame) -> p.DataFrame:
    df = df.copy()
    df["text_length"] = df[["title", "text"]].fillna("").agg(lambda x: sum(len(str(v)) for v in x), axis=1)
    logger.info(f"Longueurs calculées pour {len(df)} lignes")
    return df

def get_top_words(df: p.DataFrame, text_column="text", top_n=20, stopwords=None):
    stopwords = set(stopwords or [])
    all_text = df[text_column].dropna().str.lower().str.cat(sep=" ")
    words = [w for w in all_text.split() if w.isalpha() and w not in stopwords]
    counter = Counter(words)
    top_words = dict(counter.most_common(top_n))
    logger.info(f"Top {top_n} mots calculés")
    return top_words

def kpi_by_source(df: p.DataFrame):
    result = {}
    for source, group in df.groupby("source"):
        result[source] = {
            "count": len(group),
            "avg_text_length": round(group["text_length"].mean(), 1)
        }
    logger.info("KPIs par source calculés")
    return result

def save_top_words_csv(top_words: dict, path="reports/keywords.csv"):
    df = p.DataFrame(list(top_words.items()), columns=["word", "count"])
    df.to_csv(path, index=False)
    logger.info(f"Top mots sauvegardés dans {path}")

def update_summary_json(summary: dict, source_kpis: dict, path="reports/summary.json"):
    summary.update({"source_kpis": source_kpis})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"summary.json mis à jour avec KPIs par source")