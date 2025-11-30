import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from core.config import MODELS_DIR, TEST_SIZE, RANDOM_STATE
from core.logger import get_logger

logger = get_logger(__name__)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_classification(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    model_path = MODELS_DIR / "model.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    logger.info(f"Training done. acc={acc:.4f}, f1_macro={f1:.4f}")

    return {
        "model_path": str(model_path),
        "accuracy": float(acc),
        "f1_macro": float(f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred
    }
