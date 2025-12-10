import json
import pickle as pkl
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from core.logger import get_logger

logger = get_logger(__name__)


class Model:
    def __init__(self, model_dir, random_state, test_size, features_path, clean_data_path, summary_path):
        try:
            self.model_dir = model_dir
            self.random_state = random_state
            self.test_size = test_size
            self.X = None
            self.y = None
            self.final_result = dict()
            self.features_path = features_path
            self.clean_data_path = clean_data_path
            self.summary_path = summary_path

            self.model_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Model initialisé")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du Features: {e}")
            raise e


    def load_clean_dataframe(self) -> pd.DataFrame:
        try:
            with open(self.clean_data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"JSON chargé : {self.clean_data_path}")
        except Exception as e:
            logger.error(f"Erreur lors de la lecture du JSON {self.clean_data_path} : {e}")
            return pd.DataFrame()

        if not isinstance(data, list):
            logger.error("Le JSON chargé n'est pas une liste de lignes ! Format invalide.")
            return pd.DataFrame()

        try:
            df = pd.DataFrame(data)
            logger.info(f"DataFrame clean chargé : {len(df)} lignes")
            return df
        except Exception as e:
            logger.error(f"Erreur lors de la conversion JSON → DataFrame : {e}")
            return pd.DataFrame()


    def load_training_data(self):
        try:
            logger.info("Chargement des données d'entraînement...")

            if not self.features_path.exists():
                raise FileNotFoundError(f"Features manquantes : {self.features_path}")

            X = np.load(self.features_path)
            logger.info(f"Données d'entraînement chargées: {X.shape}")

            self.X = X

        except Exception as e:
            logger.error(f"Erreur lors du chargement des données d'entraînement : {e}")
            raise e

    def update_summary(self):
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        if self.summary_path.exists():
            with open(self.summary_path, "r", encoding="utf-8") as fh:
                summary = json.load(fh)
        else:
            summary = {}

        summary.setdefault("ml_metrics", {}).update(self.final_result)

        with open(self.summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        logger.info(f"summary.json mis à jour avec métriques ML: {self.summary_path}")


    def create_labels(self, df: pd.DataFrame):
        if "source" not in df.columns:
            raise ValueError("La colonne 'source' est manquante dans le DataFrame")
        self.y = df["source"].values


    def train_classification(self):
        try:
            logger.info("Entraînement du modèle")

            if self.X is None or self.y is None or len(self.X) == 0:
                logger.error("Données d'entraînement invalides")
                raise ValueError("Données d'entraînement invalides")

            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state, stratify=self.y)
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

            self.final_result = {
                "model_path": str(model_path),
                "accuracy": float(acc),
                "f1_macro": float(f1),
                "confusion_matrix": cm.tolist(),
                "classification_report": classification_report(y_test, y_pred, output_dict=True),
                "test_samples": len(y_test),
                "pred_samples": len(y_pred),
        }
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement du modèle: {e}")
            raise e