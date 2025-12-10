# Documentation Visualisations

## Vue d'ensemble

Le module de visualisation génère 7 figures PNG et 1 dashboard PDF à partir des données produits et métriques ML. Les visualisations utilisent des données provenant des APIs et du modèle d'apprentissage.

### Fichiers générés
- `fig1_volume_source.png` - Volume produits par source
- `fig2_top_keywords.png` - Mots-clés les plus fréquents
- `fig3_latency_distribution.png` - Distribution latence HTTP
- `fig4_http_status.png` - Distribution statuts HTTP
- `fig5_chronology.png` - Chronologie volume/temps
- `fig6_confusion_matrix.png` - Matrice confusion (ML)
- `fig7_classification_scores.png` - Scores classification (ML)
- `dashboard.pdf` - Dashboard compilé (toutes figures + page de couverture)

---

## Architecture

### Module principal : `core/viz.py`

#### Classe `Visualizer`

```python
class Visualizer:
    def __init__(self, df=None, summary_path=SUMMARY_FILE, keywords_path=None):
        """
        Initialise le visualiseur avec les données produits.
        
        Args:
            df: DataFrame pandas avec colonnes [id, title, text, category, source, 
                fetch_elapsed, fetch_status, fetch_time, ...]
            summary_path: Chemin vers summary.json (métriques ML)
            keywords_path: Chemin vers keywords.csv (top mots-clés)
        """
```

#### Flux de génération

1. **Chargement des données**
   - DataFrame produits (clean_data.json) avec métadonnées HTTP
   - Fichier summary.json pour les métriques ML
   - Fichier keywords.csv pour les mots-clés

2. **Génération des figures (1-5)**
   - Chaque méthode `plot_*()` crée une figure PNG
   - Utilise les colonnes réelles du DataFrame

3. **Génération des figures ML (6-7)**
   - Optionnelles : générées si données ML disponibles
   - Créées à partir de summary.json

4. **Compilation dashboard**
   - Agrège toutes les figures en un PDF multi-page
   - Ajoute page de couverture avec statistiques

---

## Figures détaillées

### Figure 1 : Volume par source
**Fichier** : `fig1_volume_source.png`

- **Type** : Bar chart
- **Données** : `df['source'].value_counts()`
- **Utilité** : Visualiser la distribution des produits collectés par API

### Figure 2 : Top mots-clés
**Fichier** : `fig2_top_keywords.png`

- **Type** : Horizontal bar chart (top 15)
- **Données** : `keywords.csv` (généré par Analyzer)
- **Utilité** : Identifier les termes les plus pertinents dans les descriptions

### Figure 3 : Distribution latence HTTP
**Fichier** : `fig3_latency_distribution.png`

- **Type** : Box plot + Histogram
- **Données réelles** : `df['fetch_elapsed']` (en secondes)
- **Statistiques** : 
  - Moyenne : ~1.1s
  - Min : ~0.7s
  - Max : ~1.6s
- **Groupement** : Par source API
- **Fallback** : Si fetch_elapsed absent, utilise longueur texte comme proxy

### Figure 4 : Distribution statuts HTTP
**Fichier** : `fig4_http_status.png`

- **Type** : Pie chart + Horizontal bar chart
- **Données réelles** : `df['fetch_status']` (codes HTTP)
- **Observations** : Tous les produits ont status 200 (succès)
- **Fallback** : Si fetch_status absent, utilise distribution des catégories

### Figure 5 : Chronologie volume/temps
**Fichier** : `fig5_chronology.png`

- **Type** : Line plot avec markers (ou heatmap fallback)
- **Données réelles** : `df['fetch_time']` (timestamps ISO)
- **Groupement** : Par source, par interval de 5 minutes
- **Utilité** : Voir la progression du fetch au fil du temps
- **Fallback** : Si fetch_time absent, heatmap source × catégorie

### Figure 6 : Matrice confusion (ML)
**Fichier** : `fig6_confusion_matrix.png`

- **Type** : Heatmap avec annotations
- **Données** : `summary.json['ml_metrics']['confusion_matrix']`
- **Disponibilité** : Seulement si modèle entraîné
- **Classes** : 3 (openbeautyfacts, openfoodfacts, openpetfoodfacts)

### Figure 7 : Scores classification (ML)
**Fichier** : `fig7_classification_scores.png`

- **Type** : Grouped bar chart
- **Données** : `summary.json['ml_metrics']['classification_report']`
- **Métriques par classe** : Precision, Recall, F1-score
- **Disponibilité** : Seulement si modèle entraîné

### Dashboard PDF
**Fichier** : `dashboard.pdf`

- **Nombre de pages** : 6 à 8 (selon présence données ML)
- **Page 1** : Couverture avec statistiques générales
- **Pages 2-8** : Les 7 figures

---

## Intégration dans le pipeline

### Option 5 : Visualize (autonome)

```bash
# Menu principal
Choix (1-8) : 5
# Génération des visualisations avec données disponibles
```

**Comportement** :
- Charge le DataFrame clean_data.json
- Charge les données ML du summary.json si présentes
- Génère 5+ figures selon disponibilité des données
- Génère dashboard complet

### Option 6 : Pipeline complet

```bash
Choix (1-8) : 6
# Fetch → Clean → Features → Train → Visualize
```

**Workflow** :
1. Récupère données brutes (avec `fetch_elapsed`, `fetch_status`, `fetch_time`)
2. Nettoie et préserve métadonnées HTTP
3. Extrait features
4. Entraîne modèle ML
5. **Appelle `step_visualize()`** pour générer toutes les figures

---

## Données utilisées

### Métadonnées HTTP (colonnes réelles)

| Colonne | Type | Source | Utilisation |
|---------|------|--------|-------------|
| `fetch_elapsed` | float | Fetcher.fetch() | Figure 3 (latence) |
| `fetch_status` | int | Fetcher.fetch() | Figure 4 (HTTP status) |
| `fetch_time` | str (ISO) | Fetcher.fetch() | Figure 5 (chronologie) |

**Préservation du pipeline** :
- Fetcher ajoute les colonnes
- Cleaner les préserve via `json_to_dataframe()` et `preprocess_dataframe()`
- Visualizer les utilise pour les figures

### Données ML (depuis summary.json)

| Donnée | Chemin JSON | Figure |
|--------|-------------|--------|
| Confusion matrix | `ml_metrics.confusion_matrix` | Fig 6 |
| Classification report | `ml_metrics.classification_report` | Fig 7 |
| Accuracy | `ml_metrics.accuracy` | Dashboard |
| F1 macro | `ml_metrics.f1_macro` | Dashboard |

---

## Gestion des erreurs

### Données manquantes

| Scenario | Figure | Comportement |
|----------|--------|-------------|
| `fetch_elapsed` absent | Fig 3 | Utilise longueur texte comme proxy |
| `fetch_status` absent | Fig 4 | Utilise distribution catégories |
| `fetch_time` absent | Fig 5 | Fallback : heatmap source × catégorie |
| Modèle non entraîné | Fig 6-7 | Pas générées (normal) |
| clean_data.json manquant | Toutes | Erreur → message utilisateur |

### Logs

Tous les événements sont loggés dans les fichiers logs :
- Génération réussie d'une figure : `logger.info(...)`
- Avertissement (données manquantes) : `logger.warning(...)`
- Erreur fatale : `logger.error(...)`

---

## Remarques importantes

### Cohérence
- Figures 1-5 générées même sans modèle entraîné
- Figures 6-7 générées uniquement avec données ML
- Dashboard adapte son contenu aux figures disponibles

### Configuration
- Toutes les tailles et formats sont paramétrables dans le code
- Style matplotlib : seaborn-v0_8-darkgrid
- Palettes : husl, viridis, Set2, YlOrRd selon la figure
- DPI : 300 pour qualité d'impression
