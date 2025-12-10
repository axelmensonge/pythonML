import numpy as np
import json
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from core.logger import get_logger

logger = get_logger(__name__)


class Features:
    def __init__(self, max_features, vectorizer_path, encoder_path, features_path, clean_data_path):
        try:
            self.max_features = max_features
            self.vectorizer = TfidfVectorizer(max_features=max_features)
            self.encoder = OneHotEncoder(handle_unknown="ignore")
            self.vectorizer_path = vectorizer_path
            self.encoder_path = encoder_path
            self.features_path = features_path
            self.clean_data_path = clean_data_path
            self.X = None

            self.vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
            self.encoder_path.parent.mkdir(parents=True, exist_ok=True)
            self.features_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Features initialisé")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du Features: {e}")
            raise


    def extract_features(self, df: pd.DataFrame):
        """
        Extrait TF-IDF à partir de `text_clean` et encode `category_clean`
        """
        try:
            logger.info("Extraction des features")

            if "text_clean" not in df.columns or "category_clean" not in df.columns:
                msg = "Les colonnes 'text_clean' ou 'category_clean' sont manquantes"
                logger.error(msg)
                raise ValueError(msg)

            texts = df["text_clean"].fillna("").astype(str).tolist()
            X_text = self.vectorizer.fit_transform(texts)
            logger.info(f"Text features extraites: {X_text.shape}")

            # category_clean doit être une liste par ligne — si ce n'est pas le cas, convertir
            cats = df["category_clean"].apply(lambda v: v if isinstance(v, list) else [str(v)] if pd.notna(v) else []).tolist()

            # OneHotEncoder attend une colonne 2D de strings — on convertit en string join si plusieurs labels
            cat_strings = ["|".join(row) if isinstance(row, list) else str(row) for row in cats]
            X_cat = self.encoder.fit_transform(pd.Series(cat_strings).values.reshape(-1, 1))

            # Ensure dense 2D arrays
            if hasattr(X_cat, "toarray"):
                X_cat = X_cat.toarray()
            else:
                X_cat = np.asarray(X_cat)

            if hasattr(X_text, "toarray"):
                X_text_arr = X_text.toarray()
            else:
                X_text_arr = np.asarray(X_text)

            if X_text_arr.ndim != 2:
                X_text_arr = X_text_arr.reshape(X_text_arr.shape[0], -1)
            if X_cat.ndim != 2:
                X_cat = X_cat.reshape(X_cat.shape[0], -1)

            logger.info(f"Category features extraites: {X_cat.shape}")

            X = np.hstack([X_text_arr, X_cat])
            logger.info(f"Feature matrix finale: {X.shape}")
            self.X = X
            return X

        except Exception as e:
            logger.error(f"Erreur lors de l'extraction des features: {e}")
            raise


    def save_features(self, X):
        """
        Sauvegarde la matrice X et les objets vectorizer/encoder dans des fichiers
        """
        try:
            logger.info("Sauvegarde des features")

            self.features_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.features_path, X)
            with open(self.vectorizer_path, "wb") as f:
                pickle.dump(self.vectorizer, f)
            with open(self.encoder_path, "wb") as f:
                pickle.dump(self.encoder, f)
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des features: {e}")
            raise


    def save_full_clean_normalized_dataset(self, output_path):
        """
        Sauvegarde la matrice de features normalisée (X) dans un fichier numpy
        """
        try:
            np.save(output_path, self.X)
            logger.info(f"Full clean normalized dataset sauvegardé : {output_path}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du full clean normalized dataset : {e}")
            raise


    def load_features(self):
        """
        Charge X, vectorizer et encoder depuis les fichiers
        """
        try:
            logger.info("Chargement des features")

            if not self.features_path.exists() or not self.vectorizer_path.exists() or not self.encoder_path.exists():
                msg = "Fichiers features/vectorizer/encoder manquants"
                logger.error(msg)
                raise FileNotFoundError(msg)

            X = np.load(self.features_path)
            with open(self.vectorizer_path, "rb") as f:
                text_vectorizer = pickle.load(f)
            with open(self.encoder_path, "rb") as f:
                cat_encoder = pickle.load(f)
            logger.info(f"Features chargées: {X.shape}")

            return X, text_vectorizer, cat_encoder
        except Exception as e:
            logger.error(f"Erreur lors du chargement des features: {e}")
            raise


    def load_clean_dataframe(self) -> pd.DataFrame:
        """
        Charge le JSON nettoyé (liste de dicts) et renvoie un DataFrame
        """
        try:
            if not self.clean_data_path.exists():
                msg = f"Fichier clean introuvable: {self.clean_data_path}"
                logger.error(msg)
                raise FileNotFoundError(msg)

            with open(self.clean_data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"JSON chargé : {self.clean_data_path}")
        except Exception as e:
            logger.error(f"Erreur lors de la lecture du JSON {self.clean_data_path} : {e}")
            raise

        if not isinstance(data, list):
            msg = "Le JSON chargé n'est pas une liste de lignes ! Format invalide."
            logger.error(msg)
            raise ValueError(msg)

        try:
            df = pd.DataFrame(data)
            logger.info(f"DataFrame clean chargé : {len(df)} lignes")
            return df
        except Exception as e:
            logger.error(f"Erreur lors de la conversion JSON → DataFrame : {e}")
            raise
