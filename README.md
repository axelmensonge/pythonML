Projets Python réalisé par Axel, Mathias, Quentin.B et Vincent 

Ce projet a pour but de récupérer des données à partir de 3 APIs différentes. 

Lien du répository GitHub : https://github.com/axelmensonge/pythonML.git

## Table des matières
1. [APIs utilisées](#apis-utilisées)
2. [Structure du projet](#structure-du-projet)
3. [Utilité de chaque fichier](#utilité-de-chaque-fichier)
4. [Installation et exécution](#installation-et-exécution)
5. [Pipeline de traitement](#pipeline-de-traitement)
6. [Choix ML et modèles](#choix-ml-et-modèles)
7. [Métriques et résultats](#métriques-et-résultats)
8. [Limitations et améliorations futures](#limitations-et-améliorations-futures)

---

## APIs utilisées

### 1. OpenFoodFacts
- **URL** : `https://world.openfoodfacts.net/cgi/search.pl`
- **Description** : Base de données de produits alimentaires
- **Données récupérées** : Nom du produit, ingrédients, labels, catégories, codes produits
- **Format** : JSON

### 2. OpenBeautyFacts
- **URL** : `https://world.openbeautyfacts.org/api/v2/search`
- **Description** : Base de données de produits cosmétiques et de beauté
- **Données récupérées** : Nom du produit, ingrédients, marques, catégories
- **Format** : JSON

### 3. OpenPetFoodFacts
- **URL** : `https://world.openpetfoodfacts.org/api/v2/search`
- **Description** : Base de données de produits pour animaux
- **Données récupérées** : Nom du produit, ingrédients, catégories, labels
- **Format** : JSON

---

## Structure du projet

```
pythonML/
├─ core/
│  ├─ __pycache__/
│  ├─ analyzer.py         # Analyse KPI et statistiques
│  ├─ cleaner.py          # Nettoyage et normalisation texte
│  ├─ config.py           # Configuration centralisée
│  ├─ features.py         # Extraction TF-IDF et encodage
│  ├─ fetcher.py          # Récupération depuis les APIs
│  ├─ logger.py           # Journalisation structurée
│  ├─ model.py            # Entraînement et évaluation du modèle
│  └─ viz.py              # Génération des visualisations et dashboard
│
├─ data/
│  ├─ models/             # Modèle ML + vectorizer + encoder (.pkl)
│  ├─ processed/          # Données nettoyées + features
│  └─ raw/                # Données brutes des APIs (JSON)
│
├─ reports/
│  ├─ dashboard.pdf       # Dashboard compilé avec toutes les figures
│  ├─ keywords.csv        # Top mots-clés (word, count)
│  ├─ summary.json        # Résumé complet (KPIs + métriques ML)
│  ├─ fig1_volume_source.png
│  ├─ fig2_top_keywords.png
│  ├─ fig3_latency_distribution.png
│  ├─ fig4_http_status.png
│  ├─ fig5_chronology.png
│  ├─ fig6_confusion_matrix.png
│  └─ fig7_classification_scores.png
│
├─ logs/
│  └─ marketing.log       # Journalisation complète de l'exécution
│
├─ .gitignore
├─ main.py               # Orchestration complète du pipeline
├─ README.md
└─ requirements.txt
```

---

## Utilité de chaque fichier

### core/fetcher.py
- **Rôle** : Récupération des données brutes depuis les 3 APIs
- **Entrée** : URLs des APIs, paramètres de pagination
- **Sortie** : Fichiers JSON bruts dans `/data/raw/` + métadonnées HTTP (status, latence, timestamp)

### core/cleaner.py
- **Rôle** : Nettoyage et normalisation du texte
- **Entrée** : Fichiers JSON bruts depuis `/data/raw/`
- **Sortie** : Données nettoyées dans `/data/processed/clean_data.json`
- **Opérations** :
  - Minuscules + suppression espaces superflus
  - Suppression ponctuation
  - Suppression stopwords (FR/EN)
  - Lemmatisation via spaCy
  - Fusion texte + titre
  - Encodage des catégories

### core/analyzer.py
- **Rôle** : Calcul des KPIs descriptifs et analyse statistique
- **Entrée** : Données nettoyées depuis `/data/processed/`
- **Sortie** : `summary.json` (KPIs globaux) + `keywords.csv` (top mots)
- **Métriques calculées** :
  - Total de produits par source
  - Longueur moyenne/min/max/médiane des textes
  - Distribution par source
  - Top N mots-clés avec fréquences

### core/features.py
- **Rôle** : Extraction et vectorisation des features ML
- **Entrée** : Données nettoyées
- **Sortie** : 
  - `features.npy` : Matrice X (TF-IDF + encodage catégories)
  - `vectorizer.pkl` : TF-IDF Vectorizer
  - `encoder.pkl` : OneHotEncoder pour catégories
- **Technologie** : TF-IDF (max_features=500, ngrams=1-2)

### core/model.py
- **Rôle** : Entraînement et évaluation du modèle de classification
- **Entrée** : Features + labels (source comme cible)
- **Sortie** : 
  - `model.pkl` : Modèle LogisticRegression
  - Métriques dans `summary.json`
- **Algorithme** : LogisticRegression (Scikit-learn)
- **Validation** : Train/Test split (80/20)

### core/viz.py
- **Rôle** : Génération des visualisations et dashboard
- **Entrée** : Données nettoyées + métriques ML + mots-clés
- **Sortie** : 7 figures PNG + 1 dashboard PDF
- **Figures générées** :
  1. Volume par source (barres)
  2. Top 15 mots-clés (barres horizontales)
  3. Distribution latences HTTP (box plot + histogramme)
  4. Répartition statuts HTTP (pie chart + barres)
  5. Chronologie volume/temps (line plot ou heatmap)
  6. Matrice de confusion (ML)
  7. Scores par classe (ML)

---

## Installation et exécution

### 1. Prérequis
- Python 3.9+
- pip

### 2. Installation des dépendances

```bash
pip install -r requirements.txt
```

### 3. Télécharger les modèles spaCy

```bash
python -m spacy download fr_core_news_sm
```

### 4. Lancer le pipeline

#### Option A : Menu interactif (recommandé)
```bash
python main.py
```

Le menu affiche 8 options :
```
=== Menu pipeline marketing_ml ===
1) Fetch (récupérer les données depuis les APIs)
2) Clean (nettoyer / préparer les données)
3) Features (TF-IDF + encoder)
4) Train (entraîner le modèle)
5) Visualize (générer les figures et dashboard)
6) Pipeline complet (enchaîne toutes les étapes)
7) Afficher un résumé de la pipeline
8) Quitter
```

#### Option B : Pipeline complet automatique
Choisir l'option **6** pour lancer toutes les étapes avec prompts (réutiliser les données existantes ou les recréer).

#### Option C : Exécution étape par étape
Utiliser les options **1 à 5** pour contrôler finement chaque étape du pipeline.

### 5. Récupérer les résultats

Après exécution :
- **Données nettoyées** : `data/processed/clean_data.json`
- **Modèle ML** : `data/models/model.pkl`
- **Métriques** : `reports/summary.json`
- **Figures** : `reports/fig*.png` (7 fichiers)
- **Dashboard** : `reports/dashboard.pdf`
- **Logs complets** : `logs/marketing.log`

---

## Pipeline de traitement

```
APIs (Fetch) 
    ↓
Raw Data (data/raw/)
    ↓
Clean & Normalize (cleaner.py)
    ↓
Clean Data (data/processed/clean_data.json)
    ↓
┌─→ Analyzer (KPIs + keywords)
│       ↓
│   summary.json + keywords.csv
│
├─→ Features Extraction (TF-IDF + Encoder)
│       ↓
│   features.npy + vectorizer.pkl + encoder.pkl
│       ↓
│   Model Training (LogisticRegression)
│       ↓
│   model.pkl + métriques ML
│       ↓
└─→ Visualizations (7 figures + dashboard.pdf)
        ↓
    reports/
```

---

## Choix ML et modèles

### Tâche de classification
**Objectif** : Prédire la **source** (API) d'un produit à partir de son texte nettoyé et ses catégories.

**Justification** : 
- Les trois APIs ont une structuration assez proche
- Les catégories et textes descriptifs varient selon la source

### Vectorisation des features
**TF-IDF Vectorizer** :
- `max_features=500` : Limite les dimensions pour éviter l'overfitting
- `ngrams=(1,2)` : Combine unigrammes et bigrammes pour capturer les termes composés
- `stop_words` : Français et Anglais supprimés lors du nettoyage

**Encodage des catégories** :
- `OneHotEncoder` : Encode les catégories sous forme binaire
- `handle_unknown='ignore'` : Gère les catégories non vues à l'entraînement

**Concaténation** : Hstack(TF-IDF, Catégories encodées) → Matrice X finale

### Modèle de classification
**Algorithme** : `LogisticRegression` (Scikit-learn)
- **Raison du choix** : Modèle supervisé simple, rapide et efficace pour la prédiction de la source des produits

- **Hyperparamètres** :
  - `max_iter=1000`
  - `random_state=42`

### Validation
- **Train/Test Split** : 80/20
- **Random State** : Fixé à 42

---

## Métriques et résultats

### Métriques de classification

Les métriques suivantes sont calculées et sauvegardées dans `summary.json` :

#### 1. **Accuracy**
- Définition : Proportion de prédictions correctes
- Interprétation : Métrique globale de performance
- Valeur attendue : > 0.85 (bon) / > 0.90 (très bon)

#### 2. **F1-Score (macro)**
- Définition : Moyenne harmonique de accuracy et recall (par classe)
- Interprétation : Utile pour classes déséquilibrées
- Valeur attendue : > 0.80

#### 3. **Confusion Matrix**
- Définition : Tableau croisant prédictions vs réalité
- Interprétation : Identifie les confusions entre classes
- Format : Matrice 3×3 (OpenBeauty, OpenFood, OpenPetFood)

#### 4. **Classification Report**
- Precision par classe : TP / (TP + FP)
- Recall par classe : TP / (TP + FN)
- F1 par classe : Moyenne harmonique
- Support : Nombre d'exemples par classe

### Structure de summary.json

```json
{
  "summary": {
    "total_products": 3000,
    "average_text_length": 145.5,
    "sources": {
      "openfoodfacts": 1000,
      "openbeautyfacts": 1000,
      "openpetfoodfacts": 1000
    }
  },
  "kpi_by_source": {
    "openfoodfacts": {
      "total_products": 1000,
      "avg_text_length": 150.2
    },
    ...
  },
  "ml_metrics": {
    "accuracy": 0.8950,
    "f1_macro": 0.8920,
    "test_samples": 600,
    "pred_samples": 600,
    "confusion_matrix": [[...], [...], [...]],
    "classification_report": {
      "openfoodfacts": {
        "precision": 0.90,
        "recall": 0.88,
        "f1-score": 0.89,
        "support": 200
      },
      ...
    },
    "model_path": "data/models/model.pkl"
  }
}
```

---

## Visualisations

### 7 figures obligatoires générées automatiquement :

1. **fig1_volume_source.png** : Distribution des produits par API
2. **fig2_top_keywords.png** : Top 15 mots-clés avec fréquences
3. **fig3_latency_distribution.png** : Latences HTTP (box plot + histogramme)
4. **fig4_http_status.png** : Répartition des statuts HTTP (pie chart)
5. **fig5_chronology.png** : Chronologie du volume (source vs temps)
6. **fig6_confusion_matrix.png** : Matrice de confusion du modèle
7. **fig7_classification_scores.png** : Scores (precision, recall, F1) par classe

### Dashboard PDF
- Compilation de toutes les figures dans un seul document
- Page de titre avec résumé des KPIs

---

## Limitations et améliorations futures

### Limitations actuelles

#### 1. Données
- **Volume limité** : ~3000 produits (MAX_PRODUCTS = 1000 par API)
- **Qualité variable** : Textes et catégories inconsistants selon la source
- **Langage** : Mélange de français, anglais et autres langues

#### 2. Preprocessing texte
- **Lemmatisation basique** : spaCy French n'est pas parfait pour domaines spécialisés
- **Stopwords statiques** : Ne capture pas les domaines spécifiques
- **Perte d'information** : La normalisation extrême perd les nuances

#### 3. Modèle ML
- **Feature engineering manuel** : TF-IDF + OneHot relativement basique
- **Pas d'hyperparamètres spécifiques** : LogisticRegression en configuration par défaut
- **Pas de validation croisée** : Seul un split train/test 80/20

#### 4. APIs
- **Rate limiting** : Pause de 0.2s entre requêtes, mais peut échouer avec des charges élevées
- **Pagination simple** : Pas de gestion des volumes énormes

#### 5. Reproductibilité
- **Dépendance spaCy** : Nécessite le modèle français (lourd à télécharger)

### Améliorations futures proposées

- [ ] Augmenter le volume de données (MAX_PRODUCTS = 5000+)
- [ ] Personnaliser les hyperparamètres (avec GridSearchCV)

- [ ] Tester d'autres modèles (SVM, RandomForest, GradientBoosting)
- [ ] API REST pour servir le modèle

- [ ] Deep Learning (CNN/RNN pour le texte)
- [ ] Clustering non supervisé (thématique)
- [ ] Recommandation basée sur similarité

---

## Auteurs
- Axel Vérité
- Mathias Hachani
- Quentin Boisson
- Vincent Lagarde

**Date de création** : Décembre 2025

Commande pour lancer le projet : 

```bash
python main.py
```