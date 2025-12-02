import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from core.config import MAX_FEATURES, VECTORIZER_PATH, CATEGORY_ENCODER_PATH, FEATURES_PATH
from core.logger import get_logger

logger = get_logger(__name__)


class Features:
    def __init__(self):
        self.max_features = MAX_FEATURES
        self.vectorizer = TfidfVectorizer(max_features=self.max_features)
        self.encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        self.vectorizer_path = VECTORIZER_PATH
        self.encoder_path = CATEGORY_ENCODER_PATH
        self.features_path = FEATURES_PATH

        self.vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
        self.encoder_path.parent.mkdir(parents=True, exist_ok=True)
        self.features_path.parent.mkdir(parents=True, exist_ok=True)


    def extract_features(self, df: pd.DataFrame):
        texts = df["text_clean"].fillna("").astype(str).tolist()
        X_text = self.vectorizer.fit_transform(texts).toarray()

        categories = df["category_clean"].fillna("").astype(str).tolist()
        X_cat = self.encoder.fit_transform(np.array(categories).reshape(-1, 1))

        X = np.hstack([X_text, X_cat])

        return X


    def save_features(self, X):
        np.save(self.features_path, X)

        pickle.dump(self.vectorizer, self.vectorizer_path)
        pickle.dump(self.encoder, self.encoder_path)


    def load_features(self):
        X = np.load(self.features_path)
        text_vectorizer = pickle.load(self.vectorizer_path)
        cat_encoder = pickle.load(self.encoder_path)

        return X, text_vectorizer, cat_encoder