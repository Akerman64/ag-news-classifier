import sys
from pathlib import Path

import joblib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ============================================================
# 1. CONFIGURATION DES CHEMINS
# ============================================================

# Dossier contenant index.py et preprocessing.py
API_DIR = Path(__file__).resolve().parent

# Racine du projet
ROOT_DIR = API_DIR.parent

# Chemin du modèle entraîné
MODEL_PATH = ROOT_DIR / "models" / "model.joblib"

# Permet d'importer preprocessing.py
sys.path.insert(0, str(API_DIR))

from preprocessing import supprimer_bruit, preprocesser


# ============================================================
# 2. CHARGEMENT DU MODÈLE
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Modèle introuvable : {MODEL_PATH}"
    )

# Le modèle est chargé une seule fois au démarrage
modele = joblib.load(MODEL_PATH)

CATEGORIES = {
    "World",
    "Sports",
    "Business",
    "Sci/Tech"
}

model_categories = set(modele.classes_)

if model_categories != CATEGORIES:
    raise ValueError(
        f"Catégories incorrectes dans le modèle : {model_categories}"
    )

# ============================================================
# 3. CRÉATION DE L'APPLICATION FASTAPI
# ============================================================

app = FastAPI(
    title="AG News Classifier",
    description="API de classification automatique d'articles de presse.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 4. STRUCTURE DE LA REQUÊTE
# ============================================================

class Requete(BaseModel):

    text: str = Field(
        min_length=1,
        max_length=5000,
        description="Texte de l'article à classifier"
    )

    @field_validator("text")
    @classmethod
    def verifier_texte(cls, valeur):

        valeur = valeur.strip()

        if not valeur:
            raise ValueError("Le texte ne peut pas être vide.")

        return valeur

# ============================================================
# 5. ROUTE DE VÉRIFICATION
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "message": "AG News Classifier API fonctionne correctement"
    }


# ============================================================
# 6. ROUTE DE PRÉDICTION
# ============================================================

@app.post("/api/predict")
def predict(requete: Requete):

    try:
        # 1. Récupération du texte
        texte = requete.text

        # 2. Suppression du bruit HTML et des liens
        texte = supprimer_bruit(texte)

        # 3. Preprocessing pour TF-IDF
        texte = preprocesser(texte)

        # 4. Vérification après preprocessing
        if not texte or not texte.strip():
            raise HTTPException(
                status_code=400,
                detail="Texte trop court ou sans mot porteur de sens."
            )

        # 5. Vérification de la méthode de prédiction
        if not hasattr(modele, "predict_proba"):
            raise HTTPException(
                status_code=500,
                detail="Le modèle ne permet pas de calculer les probabilités."
            )

        # 6. Prédiction
        probas = modele.predict_proba([texte])[0]

        if len(probas) != len(CATEGORIES):
            raise HTTPException(
                status_code=500,
                detail="Le nombre de probabilités retournées est incorrect."
            )

        # 7. Association catégorie / probabilité
        resultats = dict(
            zip(
                modele.classes_,
                probas.round(4).tolist()
            )
        )

        # 8. Catégorie la plus probable
        categorie = max(
            resultats,
            key=resultats.get
        )

        if categorie not in CATEGORIES:
            raise HTTPException(
                status_code=500,
                detail="Le modèle a retourné une catégorie inconnue."
            )
        
        # 9. Réponse JSON
        return {
            "categorie": categorie,
            "probabilites": resultats
        }

    except HTTPException:
        raise

    except Exception as erreur:
        print(f"Erreur pendant la prédiction : {erreur}")

        raise HTTPException(
            status_code=500,
            detail="Une erreur interne est survenue pendant la classification."
        )


# ============================================================
# 7. INTERFACE WEB
# ============================================================

# La page frontend/index.html est servie à la racine "/".
# Les routes /api/... restent prioritaires sur les fichiers de l'interface.
# Sur Vercel, ces fichiers sont copiés sur le CDN au déploiement ;
# en local (uvicorn api.index:app), c'est FastAPI qui les sert.
app.frontend("/", directory=ROOT_DIR / "frontend")