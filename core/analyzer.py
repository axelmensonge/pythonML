import json
import os
import re
from collections import Counter
from core.logger import get_logger
import pandas as pd

logger = get_logger(__name__)


class Analyzer:
    def __init__(self, summary_file, keywords_file):
        try:
            self.summary_file = summary_file
            self.keywords_file = keywords_file
            logger.info("Analyzer initialisé")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de Analyzer: {e}")
            raise e

    @staticmethod
    def compute_text_length(df: pd.DataFrame, text_column="text") -> pd.DataFrame:
        """Permet de calculer la longueur de chaque texte dans le DataFrame"""
        if text_column not in df:
            logger.error(f"Colonne '{text_column}' introuvable dans le DataFrame")
            return df

        df["text_length"] = df[text_column].fillna("").apply(len)
        logger.info(f"Longueurs calculées pour {len(df)} lignes")
        return df

    def compute_text_statistics(self, df: pd.DataFrame, text_column="text") -> dict:
        """Génération des statistiques globales sur les longueurs de tous les textes 
        (texte le plus petit, texte le plus grand, médiane, 
        écart-type et total de caractères dans tous les textes)"""
        if text_column not in df:
            logger.error(f"Colonne '{text_column}' introuvable")
            return {}

        lengths = df[text_column].fillna("").apply(len)

        stats = {
            "min_length": int(lengths.min()),  # plus petit texte
            "max_length": int(lengths.max()),  # plus grand texte
            "median_length": float(lengths.median()),  # mediane
            "std_length": float(lengths.std()),  # écart-type (Mesure la dispersion des longueurs autour de la moyenne)
            "total_chars": int(lengths.sum())  # total de caractères
        }

        logger.info("Statistiques textuelles calculées")
        return stats

    def get_top_words(self, df: pd.DataFrame, text_column="text", top_n=20):
        """Permet d’identifier les mots les plus fréquents dans les textes"""
        if text_column not in df:
            logger.error(f"Colonne '{text_column}' introuvable dans le DataFrame")
            return {}

        text_data = " ".join(df[text_column].fillna("").astype(str))

        words = re.findall(r"[a-zA-Zéèêàùûçôâ]+", text_data.lower())

        counter = Counter(words)
        top_words = dict(counter.most_common(top_n))

        logger.info(f"Top {top_n} mots calculés")
        return top_words

    def kpi_by_source(self, df: pd.DataFrame):
        """Calcule les KPI selon différentes sources"""
        if "source" not in df:
            logger.error(f"Colonne 'source' introuvable dans le DataFrame")
            return {}

        kpi = (
            df.groupby("source")
            .agg(
                total_products=("source", "count"),
                avg_text_length=("text_length", "mean")
            )
            .round(2)
            .to_dict(orient="index")
        )

        logger.info("KPIs par source calculés")
        return kpi

    def save_top_words_csv(self, top_words: dict):
        """Permet de calculer la longueur de chaque texte dans le DataFrame"""
        try:
            df = pd.DataFrame(list(top_words.items()), columns=["word", "count"])
            df.to_csv(self.keywords_file, index=False)
            logger.info(f"Top mots sauvegardés dans {self.keywords_file}")
        except Exception as e:
            logger.error(f"Erreur lors de l'export de keywords.csv : {e}")

    def update_summary_json(self, summary_data: dict, source_kpis: dict):
        """Récupère le résultat des fonctions et les enregistre dans summary.json"""
        final_payload = {
            "summary": summary_data,
            "kpi_by_source": source_kpis
        }

        try:
            summary_dir = os.path.dirname(self.summary_file)
            os.makedirs(summary_dir, exist_ok=True)
            with open(self.summary_file, "w", encoding="utf-8") as f:
                json.dump(final_payload, f, ensure_ascii=False, indent=2)
            logger.info(f"summary.json mis à jour avec KPIs par source")
        except Exception as e:
            logger.error(f"Erreur écriture summary.json : {e}")
            raise

    def run_full_analysis(self, df: pd.DataFrame) -> dict:
        """Exécute toutes les analyses et retourne un rapport complet"""
        logger.info("Démarrage de l'analyse complète")

        if "text_length" not in df.columns:
            df = self.compute_text_length(df)

        analysis_report = {
            "text_length": self.compute_text_length(df),
            "text_statistics": self.compute_text_statistics(df),
            "top_words": self.get_top_words(df, top_n=20),
            "kpi_by_source": self.kpi_by_source(df),
        }
        logger.info("Analyse complète terminée")
        return analysis_report