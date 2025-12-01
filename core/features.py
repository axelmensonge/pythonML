import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from core.config import DATA_DIR, MAX_FEATURES
from core.logger import get_logger

logger = get_logger(__name__)

VECTORIZER_PATH = DATA_DIR / "models" / "vectorizer.pkl"
FEATURES_PATH = DATA_DIR / "processed" / "features.npz"


def build_vectorizer():
    return TfidfVectorizer(max_features=MAX_FEATURES)


def extract_features(df: pd.DataFrame, text_column="text_clean"):
    texts = df[text_column].fillna("").astype(str).tolist()

    vectorizer = build_vectorizer()

    X = vectorizer.fit_transform(texts)

    return X, vectorizer


def save_features(X, vectorizer):
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    VECTORIZER_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(FEATURES_PATH, data=X.data, indices=X.indices,
                        indptr=X.indptr, shape=X.shape)

    pickle.dump(vectorizer, VECTORIZER_PATH)


def load_features():
    loader = np.load(FEATURES_PATH)

    X = np.matrix(
        (loader["data"], loader["indices"], loader["indptr"]),
        dtype=np.float64
    )

    vectorizer = pickle.load(VECTORIZER_PATH)

    return X, vectorizer