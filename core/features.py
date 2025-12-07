import numpy as np
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
            X_text = self.vectorizer.fit_transform(texts).toarray()
            logger.info(f"Text features extraites: {X_text.shape}")

            categories = df["category_clean"].fillna("").astype(str).tolist()
            X_cat = self.encoder.fit_transform(np.array(categories).reshape(-1, 1))
            logger.info(f"Category features extraites: {X_cat.shape}")

            X = np.hstack([X_text, X_cat])
            logger.info(f"Feature matrix finale: {X.shape}")

            return X
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction des features: {e}")
            raise e


    def save_features(self, X):
        try:
            logger.info("Sauvegarde des features")

            np.save(self.features_path, X)
            pickle.dump(self.vectorizer, self.vectorizer_path)
            pickle.dump(self.encoder, self.encoder_path)
        except Exception as e:
            logger.warning(f"Erreur lors de la sauvegarde des features: {e}")


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