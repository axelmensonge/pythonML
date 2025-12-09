import numpy as np
import json
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from core.logger import get_logger

logger = get_logger(__name__)


class Features:
    def __init__(self, max_features, vectorizer_path, encoder_path, features_path):
        try:
            self.max_features = max_features
            self.vectorizer = TfidfVectorizer(max_features=max_features)
            self.encoder = OneHotEncoder(handle_unknown="ignore")
            self.vectorizer_path = vectorizer_path
            self.encoder_path = encoder_path
            self.features_path = features_path
            self.X = None

            self.vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
            self.encoder_path.parent.mkdir(parents=True, exist_ok=True)
            self.features_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Features initialisé")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du Features: {e}")
            raise e


    def extract_features(self, df: pd.DataFrame):
        try:
            logger.info("Extraction des features")

            if "text_clean" not in df.columns or "category_clean" not in df.columns:
                logger.error("Les colonnes 'text_clean' ou 'category_clean' sont manquantes")
                raise ValueError("Les colonnes 'text_clean' ou 'category_clean' sont manquantes")

            texts = df["text_clean"].fillna("").astype(str).tolist()
            X_text = self.vectorizer.fit_transform(texts)
            logger.info(f"Text features extraites: {X_text.shape}")

            categories = df["category_clean"].fillna('').astype(str).tolist()
            X_cat = self.encoder.fit_transform(pd.Series(categories).values.reshape(-1, 1))
            # Assurer que X_cat est une matrice dense 2D
            if hasattr(X_cat, "toarray"):
                X_cat = X_cat.toarray()
            else:
                X_cat = np.asarray(X_cat)

            # Si X_text est sparse, densifier
            if hasattr(X_text, "toarray"):
                X_text_arr = X_text.toarray()
            else:
                X_text_arr = np.asarray(X_text)

            # Vérifier dimensions
            if X_text_arr.ndim != 2:
                X_text_arr = X_text_arr.reshape(X_text_arr.shape[0], -1)
            if X_cat.ndim != 2:
                X_cat = X_cat.reshape(X_cat.shape[0], -1)

            logger.info(f"Category features extraites: {X_cat.shape}")

            X = np.hstack([X_text_arr, X_cat])
            logger.info(f"Feature matrix finale: {X.shape}")
            self.X = X

        except Exception as e:
            logger.error(f"Erreur lors de l'extraction des features: {e}")
            raise e


    def save_features(self, X):
        try:
            logger.info("Sauvegarde des features")

            np.save(self.features_path, X)
            with open(self.vectorizer_path, "wb") as f:
                pickle.dump(self.vectorizer, f)
            with open(self.encoder_path, "wb") as f:
                pickle.dump(self.encoder, f)
        except Exception as e:
            logger.warning(f"Erreur lors de la sauvegarde des features: {e}")


    def save_full_clean_normalized_dataset(self, output_path):
        try:
            np.save(output_path, self.X)
            logger.info(f"Full clean normalized dataset sauvegardé : {output_path}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du full clean normalized dataset : {e}")


    def load_features(self):
        try:
            logger.info("Chargement des features")

            if not self.features_path.exists() or not self.vectorizer_path.exists() or not self.encoder_path.exists():
                logger.error("Fichiers features/vectorizer/encoder manquants")
                raise FileNotFoundError("Fichiers features/vectorizer/encoder manquants")

            X = np.load(self.features_path)
            text_vectorizer = pickle.load(self.vectorizer_path.open("rb"))
            cat_encoder = pickle.load(self.encoder_path.open("rb"))
            logger.info(f"Features chargées: {X.shape}")

            return X, text_vectorizer, cat_encoder
        except Exception as e:
            logger.error(f"Erreur lors du chargement des features: {e}")
            raise e

    @staticmethod
    def load_clean_dataframe(json_path: str) -> pd.DataFrame:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"JSON chargé : {json_path}")
        except Exception as e:
            logger.error(f"Erreur lors de la lecture du JSON {json_path} : {e}")
            return pd.DataFrame()

        if not isinstance(data, list):
            logger.error("Le JSON chargé n'est pas une liste de lignes ! Format invalide.")
            return pd.DataFrame()

        try:
            df = pd.DataFrame(data)
            logger.info(f"DataFrame clean chargé : {len(df)} lignes")
            return df
        except Exception as e:
            logger.error(f"Erreur lors de la conversion JSON → DataFrame : {e}")
            return pd.DataFrame()
