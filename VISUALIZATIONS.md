# Visualisations - Documentation

## Vue d'ensemble

Le module `core/viz.py` génère automatiquement 5+ figures PNG/SVG et un dashboard PDF compilé à partir des données d'analyse. Ces visualisations fournissent une vue d'ensemble complète des produits et des performances du modèle de classification.

## Figures générées

### Figure 1: Volume par source (`fig1_volume_source.png`)
**Type**: Bar chart
- **Description**: Affiche le nombre de produits par source de données
- **Données source**: `summary.json` → `sources`
- **Éléments**:
  - Barres colorées pour chaque source
  - Valeurs affichées au-dessus de chaque barre
  - Légende avec grille

### Figure 2: Top mots-clés (`fig2_top_keywords.png`)
**Type**: Horizontal bar chart
- **Description**: Les 15 mots-clés les plus fréquents avec leur compte
- **Données source**: `keywords.csv`
- **Éléments**:
  - Barre horizontale pour chaque mot
  - Gradient de couleur (viridis)
  - Fréquence affichée sur chaque barre
  - Tri décroissant

### Figure 3: Distribution des latences (`fig3_latency_distribution.png`)
**Type**: Box plot + Histogram
- **Description**: Distribution des latences de réponse par source
- **Données**: Simulées de manière réaliste
- **Éléments**:
  - Sous-figure 1: Box plot pour comparaison statistique
  - Sous-figure 2: Histogramme pour distribution complète
  - Code couleur par source

### Figure 4: Répartition des statuts HTTP (`fig4_http_status.png`)
**Type**: Pie chart + Bar chart
- **Description**: Répartition des codes de statut HTTP
- **Codes inclus**:
  - 200 OK (majorité)
  - 201 Created
  - 304 Not Modified
  - 400, 404, 500 errors
- **Éléments**:
  - Pie chart avec pourcentages
  - Bar chart avec comptes absolus

### Figure 5: Chronologie - Volume par temps (`fig5_chronology.png`)
**Type**: Line chart avec moyenne mobile
- **Description**: Évolution du volume de produits sur 30 jours
- **Éléments**:
  - Courbe principale avec marqueurs
  - Zone remplie sous la courbe
  - Moyenne mobile (7 jours)
  - Grille et légende

### Figure 6 (ML): Matrice de confusion (`fig6_confusion_matrix.png`)
**Type**: Heatmap
- **Description**: Matrice de confusion du classifieur
- **Données source**: Résultats du modèle
- **Éléments**:
  - Heatmap colorée (Blues)
  - Valeurs affichées dans les cellules
  - Labels pour classes réelles vs prédites

### Figure 6 Alt (ML): Scores par classe (`fig6_classification_scores.png`)
**Type**: Grouped bar chart
- **Description**: Scores de classification par classe
- **Métriques affichées**:
  - Precision
  - Recall
  - F1-score
- **Groupement**: Par classe

## Dashboard PDF (`dashboard.pdf`)

Compilé avec toutes les figures dans un document professionnel incluant:
- **Page 1**: Page de titre avec résumé des données
  - Total produits
  - Longueur moyenne texte
  - Détail des sources
  - Timestamp de génération
- **Pages 2-7**: Figures individuelles
- **Format**: A4 portrait (11" × 8.5")

## Utilisation

### Usage basique

```python
from core.viz import generate_visualizations

# Génère toutes les figures
viz_results = generate_visualizations()

# Avec résultats ML
viz_results = generate_visualizations(
    y_test=y_test,
    y_pred=y_pred,
    classification_report_dict=classification_report
)
```

### Usage avancé avec classe Visualizer

```python
from core.viz import Visualizer

# Initialisation
viz = Visualizer(
    summary_path="reports/summary.json",
    keywords_path="reports/keywords.csv"
)

# Génération individuelle de figures
viz.plot_volume_by_source()
viz.plot_top_keywords()
viz.plot_latency_distribution()
viz.plot_http_status_distribution()
viz.plot_chronology()

# ML figures
viz.plot_confusion_matrix(y_test, y_pred, class_names=['Classe A', 'Classe B'])
viz.plot_classification_scores(classification_report)

# Dashboard complet
viz.create_dashboard_pdf(y_test, y_pred, classification_report)
```

### Intégration dans main.py

```python
from core.viz import generate_visualizations

# Dans main()
viz_results = generate_visualizations()
print(f"✓ {len(viz_results)} fichiers générés")
```

## Caractéristiques techniques

### Configuration matplotlib
- **Style**: seaborn-v0_8-darkgrid (avec fallback)
- **Palette**: husl
- **DPI**: 300 (haute résolution)
- **Format**: PNG/SVG (PNG par défaut)
- **Colormaps**: Set2, viridis, Blues, husl

### Dépendances
```
matplotlib>=3.5.0
seaborn>=0.11.0
pandas>=1.3.0
numpy>=1.21.0
Pillow>=8.0.0  (pour PDF)
scikit-learn>=1.0.0  (pour confusion_matrix)
```

### Architecture

```
Visualizer
├── _load_summary()          # Charge summary.json
├── _load_keywords()         # Charge keywords.csv
├── plot_volume_by_source()  # Figure 1
├── plot_top_keywords()      # Figure 2
├── plot_latency_distribution()  # Figure 3
├── plot_http_status_distribution()  # Figure 4
├── plot_chronology()        # Figure 5
├── plot_confusion_matrix()  # Figure 6 (ML)
├── plot_classification_scores()  # Figure 6 Alt (ML)
├── create_dashboard_pdf()   # PDF compilé
└── generate_all_figures()   # Orchestration
```

## Fichiers de sortie

Tous les fichiers sont sauvegardés dans le répertoire `reports/`:

| Fichier | Format | Taille approx | Description |
|---------|--------|-------------|-------------|
| `fig1_volume_source.png` | PNG | ~50 KB | Volume par source |
| `fig2_top_keywords.png` | PNG | ~60 KB | Top keywords |
| `fig3_latency_distribution.png` | PNG | ~80 KB | Distribution latences |
| `fig4_http_status.png` | PNG | ~70 KB | Statuts HTTP |
| `fig5_chronology.png` | PNG | ~90 KB | Chronologie |
| `fig6_confusion_matrix.png` | PNG | ~40 KB | Matrice confusion (ML) |
| `fig6_classification_scores.png` | PNG | ~50 KB | Scores (ML) |
| `dashboard.pdf` | PDF | ~500 KB | Dashboard complet |

## Logging

Toutes les opérations sont loggées dans `logs/fetcher.log`:
- Initialisation du Visualizer
- Chargement des données
- Génération de chaque figure
- Création du PDF

Exemples:
```
2024-01-15 10:30:45 - core.viz - INFO - Visualizer initialisé
2024-01-15 10:30:46 - core.viz - INFO - Fichier summary.json chargé
2024-01-15 10:30:47 - core.viz - INFO - Figure 1 sauvegardée: reports/fig1_volume_source.png
2024-01-15 10:30:48 - core.viz - INFO - Figure 2 sauvegardée: reports/fig2_top_keywords.png
...
2024-01-15 10:31:00 - core.viz - INFO - Dashboard PDF créé: reports/dashboard.pdf
```

## Personnalisation

### Modifier les styles

```python
# Dans viz.py, section Configuration matplotlib
plt.style.use('ggplot')  # Autre style
sns.set_palette("deep")  # Autre palette
```

### Ajouter des figures personnalisées

```python
class Visualizer:
    def plot_custom_analysis(self):
        """Ma figure personnalisée"""
        fig, ax = plt.subplots(figsize=(12, 7))
        # ... code ...
        output_path = self.reports_dir / "fig_custom.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        return str(output_path)
```

## Dépannage

### Erreur: "seaborn-v0_8-darkgrid not found"
✓ Automatiquement géré avec fallback à 'default'

### Erreur: "No module named 'PIL'"
```bash
pip install Pillow
```

### Erreur: "matplotlib backend not available"
- Les backends PNG/PDF doivent être disponibles
- Installer: `pip install matplotlib --upgrade`

### Figures manquantes
- Vérifier que `summary.json` et `keywords.csv` existent
- Vérifier que le répertoire `reports/` est accessible en écriture

## Améliorations futures

- [ ] Support des figures SVG vectorielles
- [ ] Thèmes personnalisables (dark mode, light mode)
- [ ] Export Excel avec graphiques intégrés
- [ ] Génération de HTML interactif (Plotly)
- [ ] Export PowerPoint
- [ ] Animations temporelles

## Notes

- Les données de latence et statuts HTTP sont **simulées** de manière réaliste
- Pour les remplacer par des données réelles, ajouter des champs dans `summary.json`
- Le dashboard PDF utilise PIL pour convertir les PNG en PDF haute qualité
- Toutes les figures sont indépendantes et peuvent être générées individuellement
