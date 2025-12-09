import json
from pathlib import Path

import pandas as pd

from core.fetcher import Fetcher
from core.cleaner import Cleaner
from core.features import Features
from core.model import Model
from core.logger import get_logger
from core.analyzer import compute_text_length, get_top_words, kpi_by_source, save_top_words_csv, update_summary_json
from core.config import (
    RAW_DATA_DIR,
    SUMMARY_FILE,
    MAX_FEATURES,
    ENCODER_PATH,
    VECTORIZER_PATH,
    FEATURES_PATH,
    TIMEOUT,
    MAX_PRODUCTS,
    PAGE_SIZE,
    HEADERS,
    URLS,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)

logger = get_logger(__name__)


def load_raw_files(raw_dir: Path) -> dict:
    """Lit les fichiers JSON dans raw_dir et renvoie un dict name->DataFrame (brut).

    Prend en charge JSONL, JSON array, et JSON with top-level key 'products'.
    """
    dfs = {}
    if not raw_dir.exists():
        logger.warning(f"Dossier raw introuvable: {raw_dir}")
        return dfs

    for f in raw_dir.iterdir():
        if not f.is_file() or f.suffix.lower() not in (".json", ".txt"):
            continue
        try:
            df = None
            try:
                df = pd.read_json(f, lines=True)
            except ValueError:
                try:
                    df = pd.read_json(f)
                except ValueError:
                    with open(f, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    if isinstance(data, dict) and "products" in data and isinstance(data["products"], list):
                        df = pd.json_normalize(data["products"])
                    elif isinstance(data, list):
                        df = pd.json_normalize(data)
                    elif isinstance(data, dict):
                        # fallback: première liste trouvée
                        df = pd.DataFrame()
                        for k, v in data.items():
                            if isinstance(v, list):
                                df = pd.json_normalize(v)
                                break
                        if df.empty:
                            df = pd.json_normalize(data)
                    else:
                        df = pd.DataFrame()
            if df is None or df.empty:
                logger.info(f"Aucun enregistrement lu depuis {f}")
                continue
            dfs[f.stem] = df
            logger.info(f"Chargé raw file: {f} -> {len(df)} lignes")
        except Exception as e:
            logger.warning(f"Impossible de lire le fichier raw {f}: {e}")
            continue
    return dfs


def save_clean_dataframe(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(out_path, orient="records", force_ascii=False, indent=2)
    logger.info(f"Clean data sauvegardé: {out_path}")


def update_summary_ml(metrics: dict, summary_path: Path = SUMMARY_FILE) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as fh:
            summary = json.load(fh)
    else:
        summary = {}

    summary.setdefault("ml_metrics", {}).update(metrics)

    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    logger.info(f"summary.json mis à jour avec métriques ML: {summary_path}")


def run_pipeline():
    logger.info("--- Début pipeline complet : fetch -> clean -> features -> model -> summary ---")

    # instances
    features = Features(
        max_features=MAX_FEATURES,
        vectorizer_path=VECTORIZER_PATH,
        encoder_path=ENCODER_PATH,
        features_path=FEATURES_PATH,
    )

    fetcher = Fetcher(
        timeout=TIMEOUT,
        max_products=MAX_PRODUCTS,
        page_size=PAGE_SIZE,
        headers=HEADERS,
        urls=URLS,
        raw_data_dir=RAW_DATA_DIR,
    )

    cleaner = Cleaner()

    model = Model(
        model_dir=MODELS_DIR,
        random_state=RANDOM_STATE,
        test_size=TEST_SIZE,
    )

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    clean_path = processed_dir / "clean_data.json"

    # 1) Fetch
    try:
        logger.info("Étape 1/6 — fetch des APIs")
        results = fetcher.fetch_all()
        # sauvegarder raw DataFrames
        for name, df in results.items():
            out = RAW_DATA_DIR / f"{name}_all.json"
            df.to_json(out, orient="records", force_ascii=False, indent=2)
            logger.info(f"Raw sauvegardé: {out}")
    except Exception as e:
        logger.error(f"Erreur lors du fetch: {e}")
        return

    # 2) Read raw
    logger.info("Étape 2/6 — lecture des fichiers raw")
    dfs = load_raw_files(RAW_DATA_DIR)
    if not dfs:
        logger.error("Aucun fichier raw chargé — arrêt du pipeline")
        return

    # 3) Clean
    logger.info("Étape 3/6 — nettoyage")
    clean_df = cleaner.preprocess_dataframe(dfs)
    if clean_df.empty:
        logger.error("DataFrame nettoyé vide — arrêt du pipeline")
        return
    save_clean_dataframe(clean_df, clean_path)

    # 4) Analyzer KPIs
    logger.info("Étape 4/6 — calcul KPIs descriptifs")
    try:
        df_kpi = compute_text_length(clean_df)
        top_words = get_top_words(df_kpi, text_column="text_clean", top_n=30)
        save_top_words_csv(top_words)
        source_kpis = kpi_by_source(df_kpi)
        summary_base = {
            "total_products": len(df_kpi),
            "average_text_length": round(df_kpi['text_length'].mean(), 1),
            "sources": df_kpi['source'].value_counts().to_dict(),
        }
        # mise à jour initiale du summary (KPIs)
        update_summary_json(summary_base, source_kpis, path=SUMMARY_FILE)
    except Exception as e:
        logger.warning(f"Erreur KPIs descriptifs: {e}")

    # 5) Features
    logger.info("Étape 5/6 — extraction features TF-IDF + encoder")
    try:
        features.extract_features(clean_df)
        X = features.X
        features.save_features(X)
    except Exception as e:
        logger.error(f"Erreur extraction features: {e}")
        raise e

    # 6) Model training
    logger.info("Étape 6/6 — entraînement modèle")
    try:
        y = Model.create_labels(clean_df)
        model.train_classification(X, y)
        res = model.final_result
        ml_metrics = {
            "model_path": res.get("model_path"),
            "accuracy": res.get("accuracy"),
            "f1_macro": res.get("f1_macro"),
            "confusion_matrix": res.get("confusion_matrix"),
            "classification_report": res.get("classification_report"),
        }
        update_summary_ml(ml_metrics, SUMMARY_FILE)
    except Exception as e:
        logger.error(f"Erreur entraînement modèle: {e}")
        return

    logger.info("--- Pipeline terminé avec succès ---")
    print("Pipeline exécuté. Fichiers écrits: data/processed/clean_data.json, data/processed/features.npy, data/models/*, reports/summary.json")


if __name__ == "__main__":
    run_pipeline()
