import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from core.config import DATA_DIR, MAX_FEATURES
from core.logger import get_logger

logger = get_logger(__name__)

VECTORIZER_PATH = DATA_DIR / "models" / "vectorizer.pkl"
CATEGORY_ENCODER_PATH = DATA_DIR / "models" / "category_encoder.pkl"
SCALER_PATH = DATA_DIR / "models" / "scaler.pkl"
FEATURES_PATH = DATA_DIR / "processed" / "features.npy"


def build_text_vectorizer():
    return TfidfVectorizer(max_features=MAX_FEATURES)


def build_category_encoder():
    return OneHotEncoder(sparse=False, handle_unknown="ignore")


def extract_features(df: pd.DataFrame):
    texts = df["text_clean"].fillna("").astype(str).tolist()
    text_vectorizer = build_text_vectorizer()
    X_text = text_vectorizer.fit_transform(texts).toarray()  # numpy array

    categories = df["category_clean"].fillna("").astype(str).tolist()
    cat_encoder = build_category_encoder()
    X_cat = cat_encoder.fit_transform(np.array(categories).reshape(-1, 1))  # numpy array

    text_lengths = df["text_clean"].str.split().apply(len).values.reshape(-1, 1)
    scaler = StandardScaler()
    X_len = scaler.fit_transform(text_lengths)

    X = np.hstack([X_text, X_cat, X_len])

    return X, text_vectorizer, cat_encoder, scaler


def save_features(X, text_vectorizer, cat_encoder, scaler):
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    VECTORIZER_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.save(FEATURES_PATH, X)

    pickle.dump(text_vectorizer, VECTORIZER_PATH)
    pickle.dump(cat_encoder, CATEGORY_ENCODER_PATH)
    pickle.dump(scaler, SCALER_PATH)


def load_features():
    X = np.load(FEATURES_PATH)
    text_vectorizer = pickle.load(VECTORIZER_PATH)
    cat_encoder = pickle.load(CATEGORY_ENCODER_PATH)
    scaler = pickle.load(SCALER_PATH)

    return X, text_vectorizer, cat_encoder, scaler