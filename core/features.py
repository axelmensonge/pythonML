import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from core.logger import get_logger

logger = get_logger(__name__)


class Features:
    def __init__(self, max_features, vectorizer_path, encoder_path, features_path):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        self.vectorizer_path = vectorizer_path
        self.encoder_path = encoder_path
        self.features_path = features_path

        self.vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
        self.encoder_path.parent.mkdir(parents=True, exist_ok=True)
        self.features_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Features initialisé")


    def extract_features(self, df: pd.DataFrame):
        logger.info("Extraction des features")

        texts = df["text_clean"].fillna("").astype(str).tolist()
        X_text = self.vectorizer.fit_transform(texts).toarray()
        logger.info(f"Text features extraites: {X_text.shape}")

        categories = df["category_clean"].fillna("").astype(str).tolist()
        X_cat = self.encoder.fit_transform(np.array(categories).reshape(-1, 1))
        logger.info(f"Category features extraites: {X_cat.shape}")

        X = np.hstack([X_text, X_cat])
        logger.info(f"Feature matrix finale: {X.shape}")

        return X


    def save_features(self, X):
        logger.info("Sauvegarde des features")

        np.save(self.features_path, X)
        pickle.dump(self.vectorizer, self.vectorizer_path)
        pickle.dump(self.encoder, self.encoder_path)


    def load_features(self):
        logger.info("Chargement des features")

        X = np.load(self.features_path)
        text_vectorizer = pickle.load(self.vectorizer_path)
        cat_encoder = pickle.load(self.encoder_path)
        logger.info(f"Features chargées: {X.shape}")

        return X, text_vectorizer, cat_encoder