# 📰 AG News Classifier

Projet de data science IPSSI : une **web app qui classe automatiquement un article de presse** dans l'une des 4 catégories suivantes : **World, Sports, Business, Sci/Tech**.

L'utilisateur colle un texte dans l'interface, l'API interroge le modèle NLP et renvoie la catégorie prédite.

## 📌 Le projet en bref

| | |
|---|---|
| **Thématique** | Classification automatique d'articles de presse (2004-2005) selon leur rubrique |
| **Dataset** | [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) : titre + chapeau d'articles en anglais |
| **Observations** | **127 600 articles** bruts → **125 759** après nettoyage (118 334 train / 7 425 test) |
| **Catégories** | **4**, équilibrées (≈ 25 % chacune) : Sports (31 700), World (31 635), Business (31 481), Sci/Tech (30 943) |
| **Type de tâche** | Classification de texte multi-classes (apprentissage supervisé) |

---

## 🎯 Objectifs

1. Trouver un dataset de textes catégorisés → **AG News** (Hugging Face : [`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news))
2. Explorer, nettoyer et visualiser les données
3. Entraîner **plusieurs modèles NLP** et les comparer (métriques et graphes)
4. Choisir le modèle à déployer : **le plus léger** parmi les plus performants
5. Créer une **API** qui sert le modèle
6. Créer une **interface web** pour tester la classification
7. **Déployer** gratuitement (Vercel) pour qu'un tiers puisse tester

---

## 👥 Équipe et répartition

> ✏️ Remplacez A et B par vos prénoms.

| Personne | Rôle | Responsabilités | Livrables |
|---|---|---|---|
| **A : ________** | 📊 Data et modèles classiques | Exploration, nettoyage, visualisations, modèles TF-IDF, comparaison finale et choix du modèle | `01` → `03` et `05` (notebooks), `data/processed/*.csv`, `models/model.joblib` |
| **B : ________** | 🧠 Deep learning et 🌐 Application | Modèles LSTM et DistilBERT (Colab GPU), API, interface web, déploiement | `04_modeles_deep.ipynb`, `models/metrics_deep.csv`, `api/`, `public/`, déploiement Vercel |

Chacun est **responsable de ses fichiers** : on ne modifie pas le notebook de l'autre sans le prévenir (les notebooks se fusionnent très mal avec Git).

### Détail des tâches

#### 📊 Personne A : Data et modèles classiques
**Données** (`01_exploration.ipynb`, `02_preprocessing.ipynb`)
- [x] Chargement du dataset et exploration (doublons, longueurs, bruit HTML, sources)
- [x] Suppression du bruit : entités et balises HTML, publicités, liens, mentions de source
- [x] Suppression des articles aux catégories contradictoires, des doublons et de la fuite train/test
- [x] Preprocessing : minuscules, chiffres et ponctuation, stopwords (NLTK + *new*, *year*), lemmatisation
- [x] Visualisations : mots les plus fréquents et nuages de mots par catégorie
- [x] Export dans `data/processed/train_clean.csv.gz` et `test_clean.csv.gz` → **à pousser vite, B en a besoin**

**Modèles classiques** (`03_modeles_classiques.ipynb`)
- [ ] Vectorisation **TF-IDF**
- [ ] **Naive Bayes, Régression logistique, SVM linéaire, Random Forest**
- [ ] Métriques : accuracy, **F1-score macro**, matrices de confusion, temps d'entraînement, taille du modèle

**Comparaison finale** (`05_comparaison.ipynb`)
- [ ] Tableau et graphiques comparant **tous** les modèles (classiques + `models/metrics_deep.csv` de B)
- [ ] Choix argumenté du modèle à déployer (compromis performance / légèreté)
- [ ] Export du modèle retenu (pipeline TF-IDF + classifieur) dans `models/model.joblib`, avec 3 contraintes pour l'API :
  - entraîné sur la colonne **`label_text`** (le modèle renvoie directement « Sports », etc.) ;
  - doit avoir **`predict_proba`** (si c'est le SVM linéaire, l'envelopper dans `CalibratedClassifierCV`) ;
  - **moins de 50 Mo**, exporté avec la même version de scikit-learn que `requirements.txt`.

#### 🧠🌐 Personne B : Deep learning et application
**Deep learning** (`04_modeles_deep.ipynb`, sur **Colab GPU**)
- [ ] **LSTM** (embeddings entraînés sur le corpus)
- [ ] **DistilBERT** fine-tuné
- [ ] Mêmes métriques que A (accuracy, F1 macro, matrice de confusion, temps, taille), exportées dans `models/metrics_deep.csv`
- [ ] Sauvegarde des modèles deep sur **Google Drive** (trop lourds pour GitHub)

**Application** (`api/`, `public/`) : voir le [guide pas à pas](#-guide-de-lapplication-personne-b)
- [ ] `api/preprocessing.py` : même nettoyage que le notebook 02
- [ ] API Python (FastAPI) : route `POST /api/predict` qui reçoit un texte et renvoie la catégorie et les probabilités
- [ ] Interface web `public/index.html` : zone de texte, bouton « Classer », affichage du résultat et des probabilités
- [ ] Déploiement sur **Vercel** et test avec un tiers
- [ ] Mise à jour du README (lien de l'app, captures d'écran)

> 💡 B peut commencer l'API **dès le début**, avec un modèle provisoire, puis le remplacer par `models/model.joblib` quand A l'a exporté. Modèle provisoire (à lancer depuis la racine du dépôt, quelques secondes, ~3,5 Mo) :
>
> ```python
> import pandas as pd, joblib
> from sklearn.pipeline import make_pipeline
> from sklearn.feature_extraction.text import TfidfVectorizer
> from sklearn.linear_model import LogisticRegression
>
> train = pd.read_csv("data/processed/train_clean.csv.gz").dropna(subset=["text_clean"])
> modele = make_pipeline(TfidfVectorizer(max_features=50000), LogisticRegression(max_iter=1000))
> modele.fit(train["text_clean"], train["label_text"])
> joblib.dump(modele, "models/model.joblib")
> ```
>
> Le code de l'API ne change pas quand on remplace ce fichier par le modèle final.

### Format de `models/metrics_deep.csv`
Pour que A puisse intégrer les résultats de B dans la comparaison, une ligne par modèle :

```
modele,accuracy,f1_macro,temps_entrainement_s,taille_mo
LSTM,0.00,0.00,0,0
DistilBERT,0.00,0.00,0,0
```

---

## 🌐 Guide de l'application (Personne B)

### Architecture

```
Navigateur ──► public/index.html            (page statique servie par Vercel)
                    │  fetch POST /api/predict   {"text": "..."}
                    ▼
               api/index.py                 (FastAPI = fonction serverless Python sur Vercel)
                    │  supprimer_bruit() → preprocesser() → model.predict_proba()
                    ▼
               models/model.joblib          (pipeline TF-IDF + classifieur exporté par A)
```

Tout est dans **un seul dépôt et un seul projet Vercel** : la page et l'API ont la même adresse, donc pas de problème de CORS.

### Étape 1 : reproduire le preprocessing (`api/preprocessing.py`)

⚠️ **Point le plus important.** Le modèle a appris sur la colonne `text_clean`. L'API doit transformer le texte de l'utilisateur **exactement de la même façon**, sinon les prédictions seront mauvaises.

- Copier depuis `02_preprocessing.ipynb` les fonctions `supprimer_bruit`, `lemmatiser`, `preprocesser` et la liste `stop_words` (stopwords NLTK **+ `new` et `year`**).
- Les données NLTK (stopwords, wordnet) sont téléchargées une fois et **commitées** dans le dépôt, pour ne pas les télécharger à chaque démarrage sur Vercel :
  ```bash
  python -c "import nltk; nltk.download(['stopwords', 'wordnet'], download_dir='api/nltk_data')"
  ```
- En haut de `preprocessing.py` :
  ```python
  from pathlib import Path
  import nltk
  nltk.data.path.insert(0, str(Path(__file__).parent / "nltk_data"))
  ```

### Étape 2 : l'API (`api/index.py`)

```python
import sys
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).parent))
from preprocessing import supprimer_bruit, preprocesser

RACINE = Path(__file__).resolve().parent.parent
modele = joblib.load(RACINE / "models" / "model.joblib")   # chargé une seule fois

app = FastAPI(title="AG News Classifier")


class Requete(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict")
def predict(requete: Requete):
    texte = preprocesser(supprimer_bruit(requete.text))
    if not texte:
        raise HTTPException(400, "Texte trop court ou sans mot porteur de sens.")
    probas = modele.predict_proba([texte])[0]
    resultats = dict(zip(modele.classes_, probas.round(4).tolist()))
    return {"categorie": max(resultats, key=resultats.get), "probabilites": resultats}
```

Tester en local :
```bash
pip install -r requirements-dev.txt
uvicorn api.index:app --reload
```
puis ouvrir **http://127.0.0.1:8000/docs** : Swagger permet d'essayer `/api/predict` directement dans le navigateur.

### Étape 3 : l'interface (`public/index.html`)

Une seule page HTML + CSS + JavaScript, sans framework :
- une zone de texte, un bouton « Classer » et quelques exemples d'articles cliquables ;
- l'appel à l'API :
  ```javascript
  const reponse = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: zoneTexte.value }),
  });
  const { categorie, probabilites } = await reponse.json();
  ```
- l'affichage de la catégorie prédite et d'une barre par catégorie pour les probabilités ;
- un message d'erreur si l'API renvoie une erreur (texte vide, etc.).

Tester la page et l'API ensemble en local : `npx vercel dev` (nécessite Node.js), puis ouvrir http://localhost:3000.

### Étape 4 : déploiement sur Vercel

1. Ajouter à la racine **`vercel.json`**, qui envoie toutes les routes `/api/...` vers FastAPI :
   ```json
   { "rewrites": [{ "source": "/api/(.*)", "destination": "/api/index" }] }
   ```
2. Ajouter à la racine **`.vercelignore`**, pour ne pas envoyer les notebooks et les données :
   ```
   notebook/
   data/
   .venv/
   ```
3. Sur [vercel.com](https://vercel.com) : *Sign up with GitHub* → **Add New → Project** → importer le dépôt `ag-news-classifier` → *Framework Preset* : **Other** → **Deploy**.
4. Vercel installe `requirements.txt` (uniquement les dépendances de l'API) et donne une URL du type `https://ag-news-classifier.vercel.app`.
5. Chaque `push` sur `main` redéploie automatiquement ; chaque Pull Request reçoit une URL de prévisualisation.
6. Vérifier : `curl -X POST https://<url>/api/predict -H "Content-Type: application/json" -d '{"text": "Apple unveils its new iPhone"}'`

⚠️ **Limites de Vercel** (offre gratuite) :
- l'ensemble des dépendances Python doit faire **moins de 250 Mo** : c'est pour cela que `requirements.txt` ne contient que l'API (pas de pandas, jupyter, torch…) et qu'on déploie un modèle classique léger ;
- en cas de dépassement ou d'erreur au déploiement, solution de repli gratuite : **Render** ou **Hugging Face Spaces**.

---

## 🔗 Comment les parties s'enchaînent

```
            Personne A                                   Personne B
            ──────────                                   ──────────
01_exploration ──► 02_preprocessing ──► data/processed/*.csv
                                               │
                         ┌─────────────────────┴─────────────────────┐
                         ▼                                           ▼
              03_modeles_classiques                      04_modeles_deep (Colab GPU)
                         │                                           │
                         │                                 models/metrics_deep.csv
                         └─────────────────────┬─────────────────────┘
                                               ▼
                                 05_comparaison → models/model.joblib
                                               │
                                               ▼
                                 api/ (FastAPI) ◄──── public/ (interface)
                                               │
                                               ▼
                                        Déploiement Vercel
```

Chaque notebook **sauvegarde son résultat dans un fichier** que le suivant recharge : personne n'a besoin d'exécuter le notebook d'un autre pour avancer.

---

## 📁 Structure du dépôt

```
ag-news-classifier/
├── README.md
├── requirements.txt          # dépendances de l'API (installées par Vercel)
├── requirements-dev.txt      # dépendances des notebooks (en local)
├── vercel.json               # configuration du déploiement (à créer par B)
├── .vercelignore             # fichiers à ne pas déployer (à créer par B)
├── .gitignore
├── notebook/
│   ├── 01_exploration.ipynb          ✅ fait
│   ├── 02_preprocessing.ipynb        ✅ fait
│   ├── 03_modeles_classiques.ipynb
│   ├── 04_modeles_deep.ipynb
│   └── 05_comparaison.ipynb
├── data/
│   └── processed/            # données nettoyées (train_clean.csv.gz, test_clean.csv.gz)
├── models/                   # modèle léger déployé (.joblib) + metrics_deep.csv
├── api/                      # API FastAPI : index.py, preprocessing.py, nltk_data/
└── public/                   # interface web : index.html (servie telle quelle par Vercel)
```

> ⚠️ Les modèles deep learning (LSTM, DistilBERT) sont trop lourds pour GitHub : ils restent sur **Google Drive** (dossier partagé de l'équipe). Seul le modèle léger déployé va dans `models/`.

---

## 🚀 Démarrer

### 1. Récupérer le projet
```bash
git clone https://github.com/<proprietaire>/ag-news-classifier.git
cd ag-news-classifier
```

### 2. Installer les dépendances (en local)
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements-dev.txt
```

### 3. Ouvrir un notebook
- **En local** : `jupyter notebook` puis ouvrir le dossier `notebook/`
- **Sur Colab** : remplacer `github.com` par `colab.research.google.com/github` dans l'URL du notebook sur GitHub.
  Pour les modèles deep : *Exécution → Modifier le type d'exécution → GPU T4*.

---

## 🔄 Règles de travail avec Git

1. **Toujours récupérer la dernière version avant de commencer** :
   ```bash
   git checkout main
   git pull
   ```
2. **Travailler sur sa propre branche** (jamais directement sur `main`) :
   ```bash
   git checkout -b preprocessing      # ou modeles, api, interface...
   ```
3. **Enregistrer régulièrement son travail** :
   ```bash
   git add notebook/02_preprocessing.ipynb
   git commit -m "Nettoyage : suppression des doublons et des stopwords"
   git push -u origin preprocessing
   ```
4. **Ouvrir une Pull Request** sur GitHub vers `main` ; l'autre membre relit avant la fusion.
5. **Depuis Colab** : *Fichier → Enregistrer une copie dans GitHub*, en choisissant **sa branche**.

✅ Bonnes pratiques :
- Un notebook = un responsable.
- Des messages de commit clairs, en français.
- Ne jamais pousser de gros fichiers (> 50 Mo) ni de clés ou mots de passe.
- Suivre l'avancement avec les **Issues** GitHub (une issue par tâche, assignée à la bonne personne).

---

## 📊 Résultats

### Exploration et preprocessing (fait)
- **127 600 articles**, 4 catégories **parfaitement équilibrées** (30 000 par catégorie en train)
- Textes courts : **37 mots** en médiane (titre + chapeau)
- Bruit important : entités HTML dans 25 % des articles, mentions de source dans 13 %, publicités HTML
- Certaines sources sont liées à une seule catégorie (*Canadian Press* → 100 % World) → mentions supprimées
- **1 841 articles retirés** (catégories contradictoires, doublons, 156 articles communs train/test) → **125 759** au final
- Vocabulaire réduit de **188 110 à 54 450 mots** (−71 %) après preprocessing

### Comparaison des modèles
*À compléter par A (classiques) et B (deep).*

| Modèle | Accuracy | F1 macro | Temps d'entraînement | Taille |
|---|---|---|---|---|
| Naive Bayes | | | | |
| Régression logistique | | | | |
| SVM linéaire | | | | |
| Random Forest | | | | |
| LSTM | | | | |
| DistilBERT | | | | |

### Application
*À compléter par B : lien de l'app déployée et captures d'écran.*

---

## 🛠️ Stack technique

- **Données et ML** : pandas, scikit-learn, NLTK, Hugging Face `datasets` / `transformers`, PyTorch
- **Visualisation** : matplotlib, seaborn, wordcloud
- **API** : FastAPI
- **Déploiement** : Vercel
- **Collaboration** : GitHub, Google Colab, Google Drive
