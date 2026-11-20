# 📰 AG News Classifier

Projet de data science IPSSI : une **web app qui classe automatiquement un article de presse** dans l'une des 4 catégories **World, Sports, Business, Sci/Tech**.

L'utilisateur colle un texte en anglais dans l'interface, l'API le nettoie, interroge le modèle et renvoie la catégorie prédite avec la probabilité de chaque catégorie.

🔗 **Application en ligne** : *lien Vercel à ajouter après le déploiement*

## 📌 Le projet en bref

| | |
|---|---|
| **Thématique** | Classification automatique d'articles de presse (2004-2005) selon leur rubrique |
| **Dataset** | [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) : titre + chapeau d'articles en anglais |
| **Observations** | **127 600 articles** bruts → **125 759** après nettoyage (118 334 train / 7 425 test) |
| **Catégories** | **4**, équilibrées (≈ 25 % chacune) |
| **Modèle déployé** | TF-IDF + régression logistique : **F1 macro 0,909**, 7,4 Mo |
| **Stack** | pandas, NLTK, scikit-learn, PyTorch, transformers · FastAPI · HTML/JS · Vercel |

---

## 📓 Le notebook

Tout le travail de data science est dans **un seul notebook** : [`notebook/AG_News_projet_complet.ipynb`](notebook/AG_News_projet_complet.ipynb)

1. Exploration : doublons, longueurs, bruit HTML, sources
2. Nettoyage et preprocessing : suppression du bruit, des doublons et de la fuite train/test, stopwords, lemmatisation
3. Modèles classiques (TF-IDF) : Naive Bayes, régression logistique, SVM linéaire, Random Forest
4. Deep learning : LSTM bidirectionnel et DistilBERT
5. Comparaison de tous les modèles et choix du modèle déployé
6. Export du modèle et test sur de nouveaux textes

**Ouvrir sur Colab** : [colab.research.google.com/github/Akerman64/ag-news-classifier/blob/main/notebook/AG_News_projet_complet.ipynb](https://colab.research.google.com/github/Akerman64/ag-news-classifier/blob/main/notebook/AG_News_projet_complet.ipynb), puis *Exécution → Modifier le type d'exécution → GPU T4* et *Exécution → Tout exécuter* (environ 30 minutes, surtout pour DistilBERT).

Les notebooks `01` à `04` sont les versions de travail découpées par étape ; le notebook complet les regroupe.

---

## 📊 Résultats

Évaluation sur le jeu de test (7 425 articles jamais vus pendant l'entraînement).

| Modèle | Accuracy | F1 macro | Entraînement | Taille |
|---|---|---|---|---|
| DistilBERT (fine-tuné) | 0,934 | **0,934** | ~1 h | 268 Mo |
| SVM linéaire | 0,912 | 0,911 | 2 s | 7,4 Mo |
| **Régression logistique** ✅ déployée | 0,910 | 0,909 | 3 s | 7,4 Mo |
| Naive Bayes | 0,903 | 0,902 | 0,2 s | 10,6 Mo |
| LSTM bidirectionnel | 0,898 | 0,897 | ~3 min | 17 Mo |
| Random Forest | 0,891 | 0,890 | 25 s | 612 Mo |

*Temps mesurés sur un Mac (sans GPU NVIDIA) ; sur le GPU de Colab, le deep learning est plus rapide. Taille des modèles classiques = vectoriseur TF-IDF + classifieur.*

### Pourquoi on déploie un modèle classique et pas DistilBERT

DistilBERT est le meilleur modèle, mais il ne gagne que **2,2 points de F1** sur les modèles linéaires :

- **Taille** : 268 Mo pour le modèle, plus PyTorch et transformers (~750 Mo installés) → environ **1 Go**, alors que Vercel limite une fonction Python à **500 Mo**. La régression logistique fait 7,4 Mo et l'API complète environ 230 Mo.
- **Entraînement** : environ une heure contre 3 secondes, ce qui rend chaque réentraînement coûteux.
- **Réponse** : moins d'1 ms par article contre environ 9 ms pour DistilBERT sur processeur (plus de 10 fois plus lent, mais encore acceptable : ce n'est pas l'argument principal).

Entre les modèles classiques, le SVM est très légèrement meilleur mais ne donne pas de probabilités ; une fois calibré pour en donner, il descend à 0,908 et pèse 13,8 Mo. On déploie donc la **régression logistique** : F1 0,909, probabilités natives, 7,4 Mo.

---

## 🌐 L'application

```
Navigateur ──► frontend/index.html          (page web, servie à la racine /)
                    │  POST /api/predict   {"text": "..."}
                    ▼
               api/index.py                 (FastAPI)
                    │  supprimer_bruit() → preprocesser() → predict_proba()
                    ▼
               models/model.joblib          (TF-IDF + régression logistique)
```

- `api/preprocessing.py` reprend **exactement** le nettoyage du notebook : le modèle doit recevoir le texte sous la même forme qu'à l'entraînement.
- `api/nltk_data/` contient les stopwords et WordNet, versionnés dans le dépôt car Vercel ne permet pas de télécharger dans le dossier du projet.
- `entrainer_modele.py` régénère `models/model.joblib` avec la version de scikit-learn de `requirements.txt`.

### Lancer l'application en local

```bash
git clone https://github.com/Akerman64/ag-news-classifier.git
cd ag-news-classifier
python3 -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements-dev.txt
uvicorn api.index:app --reload
```

Puis ouvrir **http://127.0.0.1:8000** (l'interface) ou **http://127.0.0.1:8000/docs** (documentation interactive de l'API).

Exemple d'appel :
```bash
curl -X POST http://127.0.0.1:8000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Microsoft released a security update for Windows."}'
# {"categorie":"Sci/Tech","probabilites":{"Business":0.01,"Sci/Tech":0.98,"Sports":0.0,"World":0.01}}
```

### Déployer sur Vercel

1. Sur [vercel.com](https://vercel.com) : *Sign up with GitHub* → **Add New → Project** → importer `ag-news-classifier`.
2. *Framework Preset* : **Other**, rien d'autre à configurer → **Deploy**.
3. Vercel détecte FastAPI, lit `pyproject.toml` (`[tool.vercel] entrypoint = "api.index:app"`) et déploie l'application comme une seule fonction Python. Il n'installe que les dépendances de l'API, et copie la page `frontend/` sur son CDN. `.vercelignore` exclut les notebooks et les données.
4. Chaque `push` sur `main` redéploie automatiquement.

---

## 📁 Structure du dépôt

```
ag-news-classifier/
├── notebook/
│   ├── AG_News_projet_complet.ipynb   # tout le projet data science (à ouvrir sur Colab)
│   └── 01_… à 04_…ipynb               # versions de travail, étape par étape
├── data/processed/                    # données nettoyées (train/test .csv.gz)
├── models/
│   ├── model.joblib                   # modèle déployé (TF-IDF + régression logistique)
│   └── metrics_classiques.csv
├── api/
│   ├── index.py                       # API FastAPI (/api/predict, /api/health)
│   ├── preprocessing.py               # même nettoyage que le notebook
│   └── nltk_data/                     # stopwords + WordNet
├── frontend/index.html                # interface web (servie par app.frontend)
├── entrainer_modele.py                # régénère models/model.joblib
├── pyproject.toml, .vercelignore      # déploiement Vercel (point d'entrée de l'API)
├── requirements.txt                   # dépendances de l'API (identiques à pyproject.toml)
└── requirements-dev.txt               # dépendances notebooks + serveur local
```

---

## 👥 Équipe

| | Rôle |
|---|---|
| **Personne A** | Données (exploration, nettoyage), modèles classiques et deep learning, comparaison et choix du modèle |
| **Personne B** | API FastAPI, interface web, déploiement |

### Travailler avec Git

```bash
git checkout main && git pull          # partir de la dernière version
git checkout -b ma-branche             # une branche par sujet
git add … && git commit -m "…"         # messages clairs, en français
git push -u origin ma-branche          # puis Pull Request vers main
```
