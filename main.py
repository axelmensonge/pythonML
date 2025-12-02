import os
import json
import pandas as pd

from core.features import Features
from core.fetcher import Fetcher
from core.logger import get_logger
from core.config import RAW_DATA_DIR, SUMMARY_FILE, MAX_FEATURES, ENCODER_PATH, VECTORIZER_PATH, FEATURES_PATH, TIMEOUT, \
    MAX_PRODUCTS, PAGE_SIZE, PAGE, HEADERS, URLS, MODELS_DIR, RANDOM_STATE, TEST_SIZE
from core.analyzer import compute_text_length, get_top_words, kpi_by_source, save_top_words_csv, update_summary_json
from core.model import Model

logger = get_logger(__name__)

def read_json(path: str) -> dict:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Fichier JSON chargé : {path}")
        return data
    except FileNotFoundError:
        logger.warning(f"Fichier non trouvé: {path}")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Erreur décodage JSON: {path}")
        return {}

def load_all_products(folder: str) -> pd.DataFrame:
    all_rows = []

    for filename in os.listdir(folder):
        if not filename.endswith(".json"):
            continue

        data = read_json(os.path.join(folder, filename))
        products = data.get("products", [])

        for p in products:
            all_rows.append({
                "source": p.get("source"),
                "title": p.get("title", ""),
                "text": p.get("text", ""),
                "category": p.get("category", []),
                "published_at": p.get("published_at", None)
            })

    logger.info(f"{len(all_rows)} produits chargés dans le DataFrame.")
    return pd.DataFrame(all_rows)
    
def main():
    features = Features(
        max_features=MAX_FEATURES,
        vectorizer_path=VECTORIZER_PATH,
        encoder_path=ENCODER_PATH,
        features_path=FEATURES_PATH
    )

    fetcher = Fetcher(
        timeout=TIMEOUT,
        max_products=MAX_PRODUCTS,
        page_size=PAGE_SIZE,
        page=PAGE,
        headers=HEADERS,
        urls=URLS,
        raw_data_dir=RAW_DATA_DIR
    )

    model = Model(
        model_dir=MODELS_DIR,
        random_state=RANDOM_STATE,
        test_size=TEST_SIZE
    )

    df = load_all_products(RAW_DATA_DIR)

    if df.empty:
        print("Aucun produit trouvé dans data/raw.")
        return
    
    df = compute_text_length(df)

    top_words = get_top_words(df, text_column="text", top_n=20)
    save_top_words_csv(top_words)

    source_kpis = kpi_by_source(df)

    summary = {
        "total_products": len(df),
        "average_text_length": round(df["text_length"].mean(), 1),
        "sources": df["source"].value_counts().to_dict(),
    }

    update_summary_json(summary, source_kpis, path=SUMMARY_FILE)

    print("Génération de summary.json et keywords.csv")

if __name__ == "__main__":
    main()
