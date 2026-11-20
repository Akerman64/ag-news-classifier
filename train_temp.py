
import pandas as pd
import joblib

from pathlib import Path

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# 1. Définir les chemins
ROOT = Path(__file__).resolve().parent

TRAIN_PATH = ROOT / "data" / "processed" / "train_clean.csv.gz"
MODEL_PATH = ROOT / "models" / "model.joblib"

# 2. Charger les données nettoyées
train = pd.read_csv(TRAIN_PATH)

train = train.dropna(subset=["text_clean"])

print("Nombre d'articles :", len(train))
print("Catégories :", train["label_text"].unique())

# 3. Créer le pipeline TF-IDF + Régression logistique
modele = make_pipeline(
    TfidfVectorizer(max_features=50000),
    LogisticRegression(max_iter=1000)
)

# 4. Entraîner le modèle
print("\nEntraînement en cours...")

modele.fit(
    train["text_clean"],
    train["label_text"]
)

# 5. Créer le dossier models s'il n'existe pas
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# 6. Sauvegarder le modèle
joblib.dump(modele, MODEL_PATH)

print("\nModèle sauvegardé :", MODEL_PATH)
print("Taille :", round(MODEL_PATH.stat().st_size / 1024 / 1024, 2), "Mo")

# 7. Vérifier une prédiction
exemple = [
    "apple company launches new computer technology"
]

prediction = modele.predict(exemple)
probabilites = modele.predict_proba(exemple)[0]

print("\nCatégorie prédite :", prediction[0])

print("\nProbabilités :")

for categorie, proba in zip(modele.classes_, probabilites):
    print(f"{categorie}: {proba:.2%}")
