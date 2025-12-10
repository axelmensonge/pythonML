import json
import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords
from core.logger import get_logger
from core.config import PROCESSED_DIR

logger = get_logger(__name__)
nltk.download("stopwords")


class Cleaner:
    def __init__(self, clean_data_path):
        try:
            self.clean_data_path = clean_data_path
            self.nlp = spacy.load("fr_core_news_sm")
            self.stop_fr = set(stopwords.words("french"))
            self.stop_en = set(stopwords.words("english"))
            logger.info("Cleaner initialisé")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du Cleaner: {e}")
            raise e


    @staticmethod
    def clean_basic(text):
        if not text:
            return ""
        return text.lower().strip()


    def clean_text(self, text):
        if not text:
            logger.warning("Texte vide reçu pour le clean")
            return ""

        try:
            text = self.clean_basic(text)
            doc = self.nlp(text)

            tokens = []
            for tok in doc:
                if any(c.isdigit() for c in tok.text) or "-" in tok.text.upper():
                    tokens.append(tok.text.lower())
                    continue

                if tok.is_alpha and tok.text not in self.stop_fr and tok.text not in self.stop_en:
                    tokens.append(tok.lemma_)

            return " ".join(tokens)
        except Exception as e:
            logger.error(f"Erreur lors du clean d'un texte: {e}")
            raise e


    def clean_category(self, cat):
        if not isinstance(cat, list):
            logger.warning("Catégorie non liste reçue pour le clean")
            return ""
        if not cat:
            logger.warning("Catégorie vide reçue pour le clean")
            return ""

        try:
            cat = " ".join(cat)
            return self.clean_basic(cat)
        except Exception as e:
            logger.error(f"Erreur lors du clean de catégorie: {e}")
            raise e


    @staticmethod
    def json_to_dataframe(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"JSON chargé : {json_path}")
        except Exception as e:
            logger.error(f"Erreur lecture JSON : {e}")
            return pd.DataFrame()

        if "products" not in data or not isinstance(data["products"], list):
            logger.error("Clé 'products' manquante ou invalide dans le JSON")
            return pd.DataFrame()

        rows = []
        for item in data["products"]:
            try:
                rows.append({
                    "id": item["id"],
                    "title": item["title"],
                    "text": item["text"],
                    "category": item["category"],
                    "source": item["source"],
                })
            except Exception as e:
                logger.warning(f"Erreur lors de la lecture d’un produit : {e}")
                continue

        df = pd.DataFrame(rows)
        logger.info(f"JSON converti en DataFrame : {len(df)} lignes")

        return df


    def save_dataframe_to_json(self, df: pd.DataFrame) -> bool:
        if df is None or df.empty:
            logger.warning("Tentative d'enregistrer un DataFrame vide — sauvegarde annulée.")
            return False

        try:
            df.to_json(self.clean_data_path, orient="records", force_ascii=False, indent=4)
            logger.info(f"DataFrame sauvegardé en JSON : {self.clean_data_path}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du DataFrame en JSON : {e}")
            return False


    def preprocess_dataframe(self, dfs):
        try:
            logger.info("Process DataFrame")
            df = pd.concat(dfs.values(), ignore_index=True)

            df = df[df["id"].notna()]
            logger.info(f"Nombre de lignes après suppression des id manquants: {len(df)}")

            df["title_clean"] = df["title"].apply(self.clean_basic)
            df["text_raw_clean"] = df["text"].apply(self.clean_basic)
            df["text_merged"] = df.apply(
                lambda row: row["text_raw_clean"] if row["text_raw_clean"] else row["title_clean"],
                axis=1
            )
            df["text_clean"] = df["text_merged"].apply(self.clean_text)
            df["category_clean"] = df["category"].apply(self.clean_category)
            df = df[df["text_clean"].str.strip() != ""]
            logger.info(f"Nombre de lignes après process: {len(df)}")
            self.save_clean_data(df)

            return df
        except Exception as e:
            logger.warning(f"Erreur lors du process du dataframe: {e}")
            return pd.DataFrame()

    def save_clean_data(self, df, filename="clean_data.json"):
        try:
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            path = PROCESSED_DIR / filename
            df.to_json(path, orient="records", force_ascii=False, indent=2)
            logger.info(f"Données nettoyées sauvegardées dans {path}")
        except Exception as e:
            logger.error(f"Erreur sauvegarde données nettoyées : {e}")
