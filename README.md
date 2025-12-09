Projets Python réalisé par Axel, Mathias, Quentin.B et Vincent 

Ce projet a pour but de récupérer des données à partir de 3 APIs différentes. 

Choix de trois API différentes : 
Openfoodfacts api sur des produits alimentaires: "https://world.openfoodfacts.net/cgi/search.pl"

Openbeautyfacts api sur des produits de beautés : "https://world.openbeautyfacts.org/api/v2/search"

Openpetfoodfacts api sur des produits animaliers: "https://world.openpetfoodfacts.org/api/v2/search"


Utilité de chaque fichiers dans le dossier core :
fetcher.py -> Enregistre les données brutes stocké dans (data/raw), enregistre le nombre produit, status reponse, temps de réponse dans (summary.json)

cleaner.py -> Recupère les données brutes depuis (data/raw), enregistre les données clean dans (data/processed)

analyzer.py -> Récupère les données clean depuis (data/processed), enregistre dans summary.json et keywords.csv

features.py -> Récupère les données clean depuis (data/processed), enregistre encoder, vectorizer dans (data/processed)

model.py -> Récupère les données de features.py, enregistre accuracy, f1 score, confusion matrix, classification report dans summary.json

viz.py -> Récupère summary.json, keywords.csv et fait de belles figures



La structure du projet : 

pythonML/
├─ core/
│  ├─ __pycache__/
│  ├─ analyzer.py
│  ├─ cleaner.py
│  ├─ config.py
│  ├─ features.py
│  ├─ fetcher.py
│  ├─ logger.py
│  ├─ model.py
│  ├─ recommender.py
│  └─ viz.py
│
├─ data/
│  ├─ models/
│  ├─ processed/
│  └─ raw/
│
├─ figs/
│  └─ temp.txt
│
├─ logs/
│  ├─ fetcher.log
│  └─ temp.txt
│
├─ reports/
│  ├─ keywords.csv
│  ├─ summary.json
│  └─ temp.txt
│
├─ .gitignore
├─ main.py
├─ README.md
└─ requirements.txt


Commande pour lancer le projet : 

python -m main.py 
