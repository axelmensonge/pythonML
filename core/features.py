import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from core.config import DATA_DIR, MAX_FEATURES
from core.logger import get_logger

logger = get_logger(__name__)

VECTORIZER_PATH = DATA_DIR / "models" / "vectorizer.pkl"
CATEGORY_ENCODER_PATH = DATA_DIR / "models" / "category_encoder.pkl"
FEATURES_PATH = DATA_DIR / "processed" / "features.npy"


def build_text_vectorizer():
    return TfidfVectorizer(max_features=MAX_FEATURES)


def build_category_encoder():
    return OneHotEncoder(sparse=False, handle_unknown="ignore")


def extract_features(df: pd.DataFrame):
    texts = df["text_clean"].fillna("").astype(str).tolist()
    text_vectorizer = build_text_vectorizer()
    X_text = text_vectorizer.fit_transform(texts).toarray()

    categories = df["category_clean"].fillna("").astype(str).tolist()
    cat_encoder = build_category_encoder()
    X_cat = cat_encoder.fit_transform(np.array(categories).reshape(-1, 1))

    X = np.hstack([X_text, X_cat])

    return X, text_vectorizer, cat_encoder


def save_features(X, text_vectorizer, cat_encoder):
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    VECTORIZER_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.save(FEATURES_PATH, X)

    pickle.dump(text_vectorizer, VECTORIZER_PATH)
    pickle.dump(cat_encoder, CATEGORY_ENCODER_PATH)


def load_features():
    X = np.load(FEATURES_PATH)
    text_vectorizer = pickle.load(VECTORIZER_PATH)
    cat_encoder = pickle.load(CATEGORY_ENCODER_PATH)

    return X, text_vectorizer, cat_encoder