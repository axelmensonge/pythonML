import os
import json
import pandas as pd

from core.features import Features
from core.fetcher import fetch_all
from core.logger import get_logger
from core.config import RAW_DATA_DIR, SUMMARY_FILE, MAX_FEATURES, ENCODER_PATH, VECTORIZER_PATH, FEATURES_PATH, TIMEOUT, \
    MAX_PRODUCTS, PAGE_SIZE, PAGE, HEADERS, URLS, MODELS_DIR, RANDOM_STATE, TEST_SIZE, PROCESSED_DIR
from core.model import Model
from core.analyzer import Analyzer
from core.cleaner import Cleaner
from pandas import json_normalize

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
    all_data = []

    for filename in os.listdir(folder):
        if not filename.endswith(".json"):
            continue
        data = read_json(os.path.join(folder, filename))
        if "products" in data:
            all_data.extend(data["products"]) 

    if not all_data:
        return pd.DataFrame()
    df = json_normalize(all_data)
    logger.info(f"{len(df)} produits chargés dans le DataFrame.")
    return df

def load_clean_data() -> pd.DataFrame:
    processed_path = PROCESSED_DIR / "clean_data.json"
    if not processed_path.exists():
        logger.error("Fichier clean_data.json introuvable. Lance Cleaner d'abord.")
        return pd.DataFrame()
    
    try:
        df = pd.read_json(processed_path)
        logger.info(f"Données nettoyées chargées depuis {processed_path}")
        return df
    except Exception as e:
        logger.error(f"Erreur lecture clean_data.json : {e}")
        return pd.DataFrame()

    
def main():
    features = Features(
        max_features=MAX_FEATURES,
        vectorizer_path=VECTORIZER_PATH,
        encoder_path=ENCODER_PATH,
        features_path=FEATURES_PATH
    )

    analyzer = Analyzer()

    fetch_all() 

    model = Model(
        model_dir=MODELS_DIR,
        random_state=RANDOM_STATE,
        test_size=TEST_SIZE
    )

    df_raw = load_all_products(RAW_DATA_DIR)
    cleaner = Cleaner()
    df_clean = cleaner.preprocess_dataframe({"all": df_raw})

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    clean_path = PROCESSED_DIR / "clean_data.json"
    df_clean.to_json(clean_path, orient="records", force_ascii=False)
    logger.info(f"Données nettoyées sauvegardées dans {clean_path}")

    df = load_clean_data()
    if df.empty:
        print("Aucun donnée nettoyée trouvée.")
        return

    df = analyzer.compute_text_length(df)

    analysis_report = analyzer.run_full_analysis(df)

    analyzer.save_top_words_csv(analysis_report.get("top_words", {}))

    summary = {
        "total_products": len(df),
        "average_text_length": round(df["text_length"].mean(), 1) if "text_length" in df else 0,
        "sources": df["source"].value_counts().to_dict() if "source" in df else {},
        "text_statistics": analysis_report.get("text_statistics", {}),
    }

    analyzer.update_summary_json(
        summary_data=summary,
        source_kpis=analysis_report.get("kpi_by_source", {}),
        path=SUMMARY_FILE
    )

    print("Génération de summary.json et keywords.csv terminée.")

if __name__ == "__main__":
    main()
