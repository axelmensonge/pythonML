import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
from core.logger import get_logger
from core.config import SUMMARY_FILE, REPORTS_DIR

logger = get_logger(__name__)

# Configuration matplotlib
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    plt.style.use('default')
sns.set_palette("husl")

class Visualizer:
    def __init__(self, df=None, summary_path=SUMMARY_FILE, keywords_path=None):
        """
        Initialise le Visualizer avec le DataFrame des produits.
        
        Args:
            df: DataFrame avec colonnes [id, title, text, category, source]
            summary_path: Chemin vers summary.json
            keywords_path: Chemin vers keywords.csv
        """
        self.df = df
        self.summary_path = Path(summary_path)
        self.keywords_path = Path(keywords_path or REPORTS_DIR / "keywords.csv")
        self.reports_dir = Path(REPORTS_DIR)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.summary_data = self._load_summary()
        self.keywords_df = self._load_keywords()
        
        logger.info("Visualizer initialisé avec DataFrame" + (f" ({len(df)} lignes)" if df is not None else ""))

    def _load_summary(self) -> dict:
        """Charge les données du fichier summary.json"""
        try:
            with open(self.summary_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Fichier summary.json chargé: {self.summary_path}")
            return data
        except FileNotFoundError:
            logger.error(f"Fichier non trouvé: {self.summary_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Erreur décodage JSON: {e}")
            return {}

    def _load_keywords(self) -> pd.DataFrame:
        """Charge les données du fichier keywords.csv"""
        try:
            df = pd.read_csv(self.keywords_path)
            logger.info(f"Fichier keywords.csv chargé: {self.keywords_path}")
            return df
        except FileNotFoundError:
            logger.error(f"Fichier non trouvé: {self.keywords_path}")
            return pd.DataFrame()

    def plot_volume_by_source(self) -> str:
        """Figure 1: Volume par source (bar chart) - données réelles"""
        if self.df is None or self.df.empty:
            logger.warning("DataFrame vide pour Figure 1")
            return ""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Compter les produits par source
        sources = self.df['source'].value_counts().sort_values(ascending=False)
        
        if len(sources) == 0:
            logger.warning("Aucune source trouvée dans DataFrame")
            return ""

        colors = sns.color_palette("Set2", len(sources))
        bars = ax.bar(sources.index, sources.values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Source', fontsize=12, fontweight='bold')
        ax.set_ylabel('Nombre de produits', fontsize=12, fontweight='bold')
        ax.set_title('Volume de produits par source', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.reports_dir / "fig1_volume_source.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure 1 sauvegardée: {output_path}")
        plt.close(fig)
        return str(output_path)

    def plot_top_keywords(self) -> str:
        """Figure 2: Top mots-clés (fréquences - barres)"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        if self.keywords_df.empty:
            logger.warning("Aucune donnée de mots-clés disponible")
            return ""

        top_15 = self.keywords_df.head(15).copy()
        colors = sns.color_palette("viridis", len(top_15))
        bars = ax.barh(range(len(top_15)), top_15['count'], color=colors, edgecolor='black', linewidth=1.2)
        
        # Ajouter les valeurs à droite des barres
        for i, (bar, count) in enumerate(zip(bars, top_15['count'])):
            ax.text(count, i, f' {int(count)}', va='center', fontsize=10, fontweight='bold')
        
        ax.set_yticks(range(len(top_15)))
        ax.set_yticklabels(top_15['word'], fontsize=11)
        ax.set_xlabel('Fréquence', fontsize=12, fontweight='bold')
        ax.set_title('Top 15 mots-clés les plus fréquents', fontsize=14, fontweight='bold', pad=20)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.reports_dir / "fig2_top_keywords.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure 2 sauvegardée: {output_path}")
        plt.close(fig)
        return str(output_path)

    def plot_latency_distribution(self) -> str:
        """Figure 3: Distribution des longueurs de texte (box plot + histogram)"""
        if self.df is None or self.df.empty:
            logger.warning("DataFrame vide pour Figure 3")
            return ""
        
        # Calculer la longueur des textes
        df_temp = self.df.copy()
        df_temp['text_length'] = df_temp['text'].fillna("").apply(len)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Grouper par source pour box plot
        sources = sorted(df_temp['source'].unique())
        length_by_source = [df_temp[df_temp['source'] == src]['text_length'].values for src in sources]
        
        # Box plot
        bp = axes[0].boxplot(length_by_source, labels=sources, patch_artist=True)
        
        colors = sns.color_palette("Set2", len(sources))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[0].set_ylabel('Longueur du texte (caractères)', fontsize=12, fontweight='bold')
        axes[0].set_title('Distribution des longueurs par source (Box Plot)', fontsize=13, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Histogram global
        for source, color in zip(sources, colors):
            data = df_temp[df_temp['source'] == source]['text_length']
            axes[1].hist(data, bins=30, alpha=0.6, label=source, color=color, edgecolor='black')
        
        axes[1].set_xlabel('Longueur du texte (caractères)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Fréquence', fontsize=12, fontweight='bold')
        axes[1].set_title('Distribution des longueurs (Histogramme)', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.reports_dir / "fig3_latency_distribution.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure 3 sauvegardée: {output_path}")
        plt.close(fig)
        return str(output_path)

    def plot_http_status_distribution(self) -> str:
        """Figure 4: Distribution des catégories (top 15)"""
        if self.df is None or self.df.empty:
            logger.warning("Aucune donnée de catégories disponible")
            return ""
        
        # Utiliser category_clean s'il existe et qu'il contient des strings
        cat_col = None
        if 'category_clean' in self.df.columns:
            try:
                # Vérifier si c'est du string
                test_val = self.df['category_clean'].iloc[0]
                if isinstance(test_val, str):
                    cat_col = 'category_clean'
            except:
                pass
        
        # Sinon utiliser 'category' mais convertir les listes en strings
        if cat_col is None:
            if 'category' not in self.df.columns:
                logger.warning("Colonnes category/category_clean introuvables")
                return ""
            cat_col = 'category'
        
        df_temp = self.df.copy()
        
        # Transformer les listes en premier élément (string)
        def extract_category(val):
            if isinstance(val, list):
                return val[0] if len(val) > 0 else "Unknown"
            elif isinstance(val, str):
                return val if val else "Unknown"
            else:
                return str(val)
        
        df_temp[cat_col] = df_temp[cat_col].apply(extract_category)
        category_counts = df_temp[cat_col].value_counts().head(15)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Pie chart
        colors = sns.color_palette("husl", len(category_counts))
        labels = [cat[:12] + '...' if len(cat) > 12 else cat for cat in category_counts.index]
        wedges, texts, autotexts = axes[0].pie(
            category_counts.values, 
            labels=labels,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 9, 'weight': 'bold'}
        )
        axes[0].set_title('Top 15 Catégories (Pie Chart)', fontsize=13, fontweight='bold')
        
        # Bar chart (horizontal)
        bars = axes[1].barh(range(len(category_counts)), category_counts.values, 
                          color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Ajouter les valeurs sur les barres
        for i, (bar, count) in enumerate(zip(bars, category_counts.values)):
            axes[1].text(count, i, f' {int(count)}', va='center', fontsize=9, fontweight='bold')
        
        axes[1].set_yticks(range(len(category_counts)))
        labels_y = [cat[:25] + '...' if len(cat) > 25 else cat for cat in category_counts.index]
        axes[1].set_yticklabels(labels_y, fontsize=9)
        axes[1].set_xlabel('Nombre de produits', fontsize=11, fontweight='bold')
        axes[1].set_title('Top 15 Catégories (Barres)', fontsize=13, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.reports_dir / "fig4_http_status.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure 4 sauvegardée: {output_path}")
        plt.close(fig)
        return str(output_path)

    def plot_chronology(self) -> str:
        """Figure 5: Distribution source × catégorie (heatmap)"""
        if self.df is None or self.df.empty or 'source' not in self.df.columns:
            logger.warning("Aucune donnée pour heatmap source/catégorie")
            return ""
        
        # Copier le DataFrame et préparer les colonnes
        df_temp = self.df.copy()
        
        # Utiliser category_clean s'il existe et qu'il contient des strings
        cat_col = None
        if 'category_clean' in df_temp.columns:
            try:
                test_val = df_temp['category_clean'].iloc[0]
                if isinstance(test_val, str):
                    cat_col = 'category_clean'
            except:
                pass
        
        if cat_col is None:
            cat_col = 'category'
        
        # Transformer les listes en premier élément (string)
        def extract_category(val):
            if isinstance(val, list):
                return val[0] if len(val) > 0 else "Unknown"
            elif isinstance(val, str):
                return val if val else "Unknown"
            else:
                return str(val)
        
        df_temp[cat_col] = df_temp[cat_col].apply(extract_category)
        
        # Créer une table de contingence (top 12 catégories pour lisibilité)
        top_cats = df_temp[cat_col].value_counts().head(12).index
        df_filtered = df_temp[df_temp[cat_col].isin(top_cats)]
        contingency_table = pd.crosstab(df_filtered['source'], df_filtered[cat_col])
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Heatmap
        sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlOrRd', 
                   cbar_kws={'label': 'Nombre de produits'}, ax=ax, 
                   linewidths=0.5, linecolor='gray')
        
        ax.set_xlabel('Catégorie (Top 12)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Source', fontsize=12, fontweight='bold')
        ax.set_title('Matrice Source × Catégorie', fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        
        plt.tight_layout()
        output_path = self.reports_dir / "fig5_chronology.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure 5 sauvegardée: {output_path}")
        plt.close(fig)
        return str(output_path)

    def plot_confusion_matrix(self, y_test, y_pred, class_names=None) -> str:
        """Figure 6 (ML): Matrice de confusion pour classification"""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_test, y_pred)
        
        if class_names is None:
            class_names = [f'Classe {i}' for i in range(len(np.unique(y_test)))]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Heatmap
        im = ax.imshow(cm, cmap='Blues', aspect='auto', interpolation='nearest')
        
        # Ajouter les valeurs dans les cellules
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text = ax.text(j, i, cm[i, j], ha="center", va="center", 
                             color="white" if cm[i, j] > cm.max() / 2 else "black",
                             fontsize=12, fontweight='bold')
        
        # Labels et titre
        ax.set_xlabel('Prédictions', fontsize=12, fontweight='bold')
        ax.set_ylabel('Réalité', fontsize=12, fontweight='bold')
        ax.set_title('Matrice de confusion - Classification', fontsize=14, fontweight='bold', pad=20)
        
        # Ticks
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Nombre de samples', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        output_path = self.reports_dir / "fig6_confusion_matrix.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure 6 (ML) sauvegardée: {output_path}")
        plt.close(fig)
        return str(output_path)

    def plot_classification_scores(self, classification_report_dict) -> str:
        """Figure 6 Alt (ML): Scores par classe (barres)"""
        # Extraire les métriques par classe
        metrics = ['precision', 'recall', 'f1-score']
        classes = [k for k in classification_report_dict.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        x = np.arange(len(classes))
        width = 0.25
        colors = sns.color_palette("Set2", len(metrics))
        
        for i, metric in enumerate(metrics):
            values = [classification_report_dict[cls][metric] for cls in classes]
            ax.bar(x + i * width, values, width, label=metric, color=colors[i], 
                  alpha=0.8, edgecolor='black', linewidth=1.2)
        
        ax.set_xlabel('Classe', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Scores de classification par classe', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x + width)
        ax.set_xticklabels(classes, fontsize=11)
        ax.legend(fontsize=11)
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.reports_dir / "fig6_classification_scores.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure 6 (ML - Scores) sauvegardée: {output_path}")
        plt.close(fig)
        return str(output_path)

    def create_dashboard_pdf(self, y_test=None, y_pred=None, classification_report_dict=None) -> str:
        """Crée un dashboard PDF avec toutes les figures"""
        try:
            from matplotlib.backends.backend_pdf import PdfPages
        except ImportError:
            logger.warning("matplotlib PDF backend non disponible. Installation recommandée.")
            return ""
        
        pdf_path = self.reports_dir / "dashboard.pdf"
        
        with PdfPages(pdf_path) as pdf:
            # Page de titre
            fig = plt.figure(figsize=(11, 8.5))
            fig.suptitle('Dashboard - Analyse des Produits', fontsize=24, fontweight='bold', y=0.9)
            
            ax = fig.add_subplot(111)
            ax.axis('off')
            
            info_text = f"""
            Résumé de l'analyse
            
            Nombre total de produits: {self.summary_data.get('summary', {}).get('total_products', 'N/A')}
            Longueur moyenne de texte: {self.summary_data.get('summary', {}).get('average_text_length', 'N/A')} caractères
            
            Sources:
            """
            
            sources = self.summary_data.get('summary', {}).get('sources', {})
            for source, count in sources.items():
                info_text += f"\n  • {source}: {count} produits"
            
            info_text += f"""
            
            Généré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            ax.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center',
                   family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Page 1: Volume par source
            self._add_figure_to_pdf(pdf, "fig1_volume_source.png")
            
            # Page 2: Top keywords
            self._add_figure_to_pdf(pdf, "fig2_top_keywords.png")
            
            # Page 3: Latency distribution
            self._add_figure_to_pdf(pdf, "fig3_latency_distribution.png")
            
            # Page 4: HTTP status
            self._add_figure_to_pdf(pdf, "fig4_http_status.png")
            
            # Page 5: Chronology
            self._add_figure_to_pdf(pdf, "fig5_chronology.png")
            
            # Page 6: Classification ML
            if y_test is not None and y_pred is not None:
                self._add_figure_to_pdf(pdf, "fig6_confusion_matrix.png")
            
            # Page 7: Classification scores (si disponible)
            if classification_report_dict is not None:
                self._add_figure_to_pdf(pdf, "fig6_classification_scores.png")
        
        logger.info(f"Dashboard PDF créé: {pdf_path}")
        return str(pdf_path)

    def _add_figure_to_pdf(self, pdf, figure_name: str):
        """Ajoute une figure au PDF"""
        fig_path = self.reports_dir / figure_name
        if fig_path.exists():
            img = plt.imread(fig_path)
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111)
            ax.imshow(img)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    def generate_all_figures(self, y_test=None, y_pred=None, classification_report_dict=None):
        """Génère toutes les figures"""
        logger.info("Génération de toutes les figures...")
        
        paths = {
            'fig1_volume_source': self.plot_volume_by_source(),
            'fig2_top_keywords': self.plot_top_keywords(),
            'fig3_latency_distribution': self.plot_latency_distribution(),
            'fig4_http_status': self.plot_http_status_distribution(),
            'fig5_chronology': self.plot_chronology(),
        }
        
        if y_test is not None and y_pred is not None:
            paths['fig6_confusion_matrix'] = self.plot_confusion_matrix(y_test, y_pred)
        
        if classification_report_dict is not None:
            paths['fig6_classification_scores'] = self.plot_classification_scores(classification_report_dict)
        
        dashboard_path = self.create_dashboard_pdf(y_test, y_pred, classification_report_dict)
        paths['dashboard'] = dashboard_path
        
        logger.info(f"Génération de figures terminée. Total: {len(paths)} fichiers")
        return paths


# Fonction utilitaire pour intégration dans main.py
def generate_visualizations(df=None, y_test=None, y_pred=None, classification_report_dict=None):
    """
    Fonction principale pour générer toutes les visualisations.
    
    Args:
        df: DataFrame avec colonnes [id, title, text, category, source]
        y_test: Labels de test (pour ML)
        y_pred: Prédictions (pour ML)
        classification_report_dict: Rapport de classification sklearn (pour ML)
    
    Returns:
        Dict avec chemins des fichiers générés
    """
    viz = Visualizer(df=df)
    return viz.generate_all_figures(y_test, y_pred, classification_report_dict)
