import pickle as pkl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from core.config import MODELS_DIR, TEST_SIZE, RANDOM_STATE
from core.logger import get_logger

logger = get_logger(__name__)


class Model:
    def __init__(self):
        self.model_dir = MODELS_DIR
        self.random_state = RANDOM_STATE
        self.test_size = TEST_SIZE

        self.model_dir.mkdir(parents=True, exist_ok=True)

    def train_classification(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state, stratify=y)

        model = LogisticRegression(max_iter=1000, random_state=self.random_state)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        model_path = self.model_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pkl.dump(model, f)
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
