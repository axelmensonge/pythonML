import pickle as pkl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from core.logger import get_logger

logger = get_logger(__name__)


class Model:
    def __init__(self, model_dir, random_state, test_size):
        try:
            self.model_dir = model_dir
            self.random_state = random_state
            self.test_size = test_size

            self.model_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Model initialisé")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du Features: {e}")
            raise e


    def train_classification(self, X, y):
        try:
            logger.info("Entraînement du modèle")

            if X is None or y is None or len(X) == 0:
                logger.error("Données d'entraînement invalides")
                raise ValueError("Données d'entraînement invalides")

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state, stratify=y)
            logger.info(f"Split des données: X_train={X_train.shape}, X_test={X_test.shape}")

            model = LogisticRegression(max_iter=1000, random_state=self.random_state)
            model.fit(X_train, y_train)
            logger.info("Entraînement terminé")

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            cm = confusion_matrix(y_test, y_pred)

            model_path = self.model_dir / "model.pkl"
            with open(model_path, "wb") as f:
                pkl.dump(model, f)

            logger.info(f"Modèle sauvegarder dans {model_path}")
            logger.info(f"Métriques des modèles - Accuracy: {acc:.4f}, F1 macro: {f1:.4f}")

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
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement du modèle: {e}")
            raise e