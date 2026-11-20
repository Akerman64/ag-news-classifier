"""Entraîne le modèle déployé dans l'API et le sauvegarde dans models/model.joblib.

Modèle retenu dans le notebook AG_News_projet_complet.ipynb (partie 6) :
TF-IDF (mots + paires de mots) + régression logistique (C = 1).

On le régénère avec ce script plutôt que depuis Colab pour que la version de
scikit-learn soit celle de requirements.txt (un modèle joblib ne se recharge
pas forcément avec une autre version).

Usage, depuis la racine du dépôt :
    pip install -r requirements-dev.txt
    python entrainer_modele.py
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

RACINE = Path(__file__).resolve().parent
DONNEES = RACINE / "data" / "processed"
CHEMIN_MODELE = RACINE / "models" / "model.joblib"

train = pd.read_csv(DONNEES / "train_clean.csv.gz")
test = pd.read_csv(DONNEES / "test_clean.csv.gz")
print(f"Train : {len(train)} articles | Test : {len(test)} articles")

modele = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100_000, sublinear_tf=True),
    LogisticRegression(C=1, max_iter=1000),
)
modele.fit(train["text_clean"], train["label_text"])

y_pred = modele.predict(test["text_clean"])
print(f"Accuracy : {accuracy_score(test['label_text'], y_pred):.4f}")
print(f"F1 macro : {f1_score(test['label_text'], y_pred, average='macro'):.4f}")

CHEMIN_MODELE.parent.mkdir(exist_ok=True)
joblib.dump(modele, CHEMIN_MODELE)
print(f"Modèle sauvegardé : {CHEMIN_MODELE.relative_to(RACINE)} ({CHEMIN_MODELE.stat().st_size / 1e6:.1f} Mo)")
