import json

from core.fetcher import Fetcher
from core.cleaner import Cleaner
from core.features import Features
from core.model import Model
from core.analyzer import Analyzer
from core.logger import get_logger
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
    KEYWORDs_FILE,
    CLEANED_DATA_DIR,
)

logger = get_logger(__name__)


def prompt_yes_no(prompt: str, default: str = "o") -> bool:
    ans = input(f"{prompt} [{default}/n] : ").strip().lower()
    if ans == "" or ans == default:
        return True
    return ans in ("o", "oui", "y", "yes")

def step_fetch(fetcher: Fetcher, force: bool = False):
    try:
        if not force:
            reuse = prompt_yes_no("Réutiliser les fichiers bruts déjà présents dans data/raw ?")
            if reuse:
                print("Utilisation des fichiers raw existants (fetch SKIPPED).")
                logger.info("Fetch skipped: reuse raw files")
                return

        print("Lancement du fetch depuis les APIs...")
        fetcher.fetch_all()
        print("Fetch terminé.")
    except Exception as e:
        logger.exception(f"Erreur durant fetch: {e}")
        print(f"Erreur pendant le fetch: {e}")


def step_clean(cleaner: Cleaner, force: bool = False):
    try:
        if not force and CLEANED_DATA_DIR.exists():
            reuse = prompt_yes_no(f"Un fichier clean existe déjà ({CLEANED_DATA_DIR}). Le réutiliser ?")
            if reuse:
                print("Réutilisation du clean existant.")
                logger.info("Clean step skipped: reuse existing clean file")
                return

        candidates = {
            "openbeautyfacts": RAW_DATA_DIR / "openbeautyfacts_all_response.json",
            "openpetfoodfacts": RAW_DATA_DIR / "openpetfoodfacts_all_response.json",
            "openfoodfacts": RAW_DATA_DIR / "openfoodfacts_all_response.json",
        }

        dfs = {}
        for key, path in candidates.items():
            if path.exists():
                try:
                    df = cleaner.json_to_dataframe(path)
                    dfs[key] = df
                    logger.info(f"Chargé raw pour {key}: {len(df)} lignes")
                except Exception as e:
                    logger.exception(f"Erreur lecture raw {path}: {e}")
                    print(f"Erreur lecture {path}: {e}")
            else:
                logger.warning(f"Fichier raw attendu non trouvé: {path}")

        if not dfs:
            print("Aucun fichier raw valable trouvé. Lancez d'abord le fetch.")
            return

        try:
            clean_df = cleaner.preprocess_dataframe(dfs)
        except Exception as e:
            logger.exception(f"Erreur preprocess_dataframe: {e}")
            print(f"Erreur lors du preprocessing: {e}")
            return

        try:
            cleaner.save_dataframe_to_json(clean_df)
            print(f"Données nettoyées sauvegardées : {CLEANED_DATA_DIR}")
        except Exception as e:
            logger.exception(f"Erreur sauvegarde clean: {e}")
            print(f"Erreur lors de la sauvegarde du clean: {e}")

    except Exception as e:
        logger.exception(f"Erreur dans step_clean: {e}")
        print(f"Erreur inattendue dans l'étape clean: {e}")


def step_features(features: Features, force: bool = False):
    try:
        exist_feats = FEATURES_PATH.exists()
        exist_vec = VECTORIZER_PATH.exists()
        exist_enc = ENCODER_PATH.exists()

        if not force and exist_feats and exist_vec and exist_enc:
            reuse = prompt_yes_no("Des features / vectorizer / encoder existent. Les réutiliser ?")
            if reuse:
                try:
                    X, vec, enc = features.load_features()
                    features.vectorizer = vec
                    features.encoder = enc
                    features.X = X
                    print(f"Features chargées : {X.shape}")
                    return
                except Exception as e:
                    logger.exception(f"Échec chargement features existantes: {e}")
                    print("Impossible de charger les artefacts existants ; on recrée.")

        try:
            clean_df = features.load_clean_dataframe()
        except Exception as e:
            logger.exception(f"Impossible de charger clean_data pour features: {e}")
            print(f"Fichier clean manquant ou invalide: {e}")
            return

        try:
            features.extract_features(clean_df)
            features.save_features(features.X)
            print("Features extraites et sauvegardées.")
        except Exception as e:
            logger.exception(f"Erreur extraction/sauvegarde features: {e}")
            print(f"Erreur lors de l'extraction des features: {e}")

    except Exception as e:
        logger.exception(f"Erreur dans step_features: {e}")
        print(f"Erreur inattendue dans l'étape features: {e}")


def step_train(model: Model, features: Features, force: bool = False):
    try:
        model_path = MODELS_DIR / "model.pkl"
        if not force and model_path.exists():
            reuse = prompt_yes_no(f"Un modèle existe déjà ({model_path}). Le réutiliser (pas de réentraînement) ?")
            if reuse:
                print("Réutilisation du modèle existant.")
                return

        try:
            X, _, _ = features.load_features()
        except Exception as e:
            logger.exception(f"Features non chargées: {e}")
            print("Features manquantes. Exécutez d'abord l'étape features.")
            return

        try:
            df_clean = model.load_clean_dataframe()
        except Exception as e:
            logger.exception(f"Clean data non chargée pour entraînement: {e}")
            print("Clean data manquante. Exécutez d'abord l'étape clean.")
            return

        try:
            model.X = X
            model.create_labels(df_clean)
            model.train_classification()
            model.update_summary()
            print("Entraînement terminé ; métriques sauvegardées dans summary.json")
        except Exception as e:
            logger.exception(f"Erreur lors de l'entraînement: {e}")
            print(f"Erreur lors de l'entraînement: {e}")

    except Exception as e:
        logger.exception(f"Erreur dans step_train: {e}")
        print(f"Erreur inattendue dans l'étape train: {e}")

def print_summary(summary_path):
    if not summary_path.exists():
        print("summary.json introuvable.")
        return
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("\n=== SUMMARY ===")
    print(f"Total produits : {data['summary']['total_products']}")
    print(f"Longueur moyenne du texte : {data['summary']['average_text_length']:.1f}")
    print("Produits par source :")
    for src, count in data['summary']['sources'].items():
        print(f"  - {src} : {count}")

    print("\n=== KPI PAR SOURCE ===")
    for src, kpi in data['kpi_by_source'].items():
        print(f"  {src}: total={kpi['total_products']}, avg_length={kpi['avg_text_length']:.2f}")

    if "ml_metrics" in data:
        ml = data["ml_metrics"]
        print("\n=== ML METRICS ===")
        print(f"Accuracy : {ml.get('accuracy', 0):.4f}")
        print(f"F1 macro : {ml.get('f1_macro', 0):.4f}")
        print(f"Test samples : {ml.get('test_samples', 'N/A')}")
        print(f"Pred samples : {ml.get('pred_samples', 'N/A')}")
        print(f"Modèle : {ml.get('model_path', 'N/A')}")
    print("=================\n")

def do_pipeline(fetcher, cleaner, features, model, analyzer):
    print("Pipeline complet — on te posera des choix pour réutiliser ou recréer les artefacts.")
    try:
        if prompt_yes_no("Réutiliser data/raw existant ?"):
            logger.info("Pipeline: reuse raw")
        else:
            step_fetch(fetcher, force=True)

        if prompt_yes_no("Réutiliser clean existant ?"):
            logger.info("Pipeline: reuse clean")
        else:
            step_clean(cleaner, force=True)

        # Analyzer KPIs
        try:
            clean_df = features.load_clean_dataframe()
            df_kpi = analyzer.compute_text_length(clean_df, text_column="text_clean")
            top_words = analyzer.get_top_words(df_kpi, text_column="text_clean", top_n=30)
            analyzer.save_top_words_csv(top_words)
            source_kpis = analyzer.kpi_by_source(df_kpi)
            summary_base = {
                "total_products": len(df_kpi),
                "average_text_length": round(df_kpi['text_length'].mean(), 1),
                "sources": df_kpi['source'].value_counts().to_dict(),
            }
            analyzer.update_summary_json(summary_base, source_kpis)
            print("KPIs descriptifs calculés.")
        except Exception as e:
            logger.exception(f"Erreur KPIs dans pipeline: {e}")
            print(f"Erreur calcul KPIs: {e}")

        if prompt_yes_no("Réutiliser features/vectorizer/encoder existants ?"):
            try:
                X, vec, enc = features.load_features()
                features.vectorizer = vec
                features.encoder = enc
                features.X = X
                print("Features existantes chargées.")
            except Exception:
                print("Impossible de charger les features existantes ; on les recrée.")
                step_features(features, force=True)
        else:
            step_features(features, force=True)

        if prompt_yes_no("Réutiliser modèle existant si présent ?"):
            model_path = MODELS_DIR / "model.pkl"
            if not model_path.exists():
                print("Aucun modèle trouvé ; entraînement nécessaire.")
                step_train(model, features, force=True)
        else:
            step_train(model, features, force=True)

        print("Pipeline complet terminé.")
    except Exception as e:
        logger.exception(f"Erreur inattendue dans pipeline complet: {e}")
        print(f"Erreur inattendue pendant le pipeline: {e}")


def main_menu():
    features = Features(
        max_features=MAX_FEATURES,
        vectorizer_path=VECTORIZER_PATH,
        encoder_path=ENCODER_PATH,
        features_path=FEATURES_PATH,
        clean_data_path=CLEANED_DATA_DIR,
    )

    analyzer = Analyzer(summary_file=SUMMARY_FILE, keywords_file=KEYWORDs_FILE)

    fetcher = Fetcher(
        timeout=TIMEOUT,
        max_products=MAX_PRODUCTS,
        page_size=PAGE_SIZE,
        headers=HEADERS,
        urls=URLS,
        raw_data_dir=RAW_DATA_DIR,
    )

    cleaner = Cleaner(clean_data_path=CLEANED_DATA_DIR)

    model = Model(
        model_dir=MODELS_DIR,
        random_state=RANDOM_STATE,
        test_size=TEST_SIZE,
        features_path=FEATURES_PATH,
        clean_data_path=CLEANED_DATA_DIR,
        summary_path=SUMMARY_FILE,
    )

    while True:
        print("\n=== Menu pipeline marketing_ml ===")
        print("1) Fetch (récupérer les données depuis les APIs)")
        print("2) Clean (nettoyer / préparer les données)")
        print("3) Features (TF-IDF + encoder)")
        print("4) Train (entraîner le modèle)")
        print("5) Pipeline complet (enchaîne toutes les étapes)")
        print("6) Afficher un résumé de la pipeline")
        print("7) Quitter")
        choice = input("Choix (1-7) : ").strip()

        try:
            if choice == "1":
                step_fetch(fetcher)
            elif choice == "2":
                step_clean(cleaner)
            elif choice == "3":
                step_features(features)
            elif choice == "4":
                step_train(model, features)
            elif choice == "5":
                do_pipeline(fetcher, cleaner, features, model, analyzer)
            elif choice == "6":
                print_summary(SUMMARY_FILE)
            elif choice == "7":
                print("Quitter...")
                break
            else:
                print("Choix invalide, entrez un chiffre entre 1 et 7.")
        except Exception as e:
            logger.exception(f"Erreur lors de l'exécution de l'option {choice}: {e}")
            print(f"Erreur lors de l'exécution : {e}")


if __name__ == "__main__":
    main_menu()
