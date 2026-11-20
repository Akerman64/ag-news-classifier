
import re
import html

from pathlib import Path
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ============================================================
# 1. CONFIGURATION NLTK
# ============================================================

API_DIR = Path(__file__).resolve().parent

# Ressources NLTK versionnées dans le dépôt (stopwords + wordnet) :
# sur Vercel, le dossier du projet est en lecture seule, on ne peut pas y télécharger.
NLTK_DIR = API_DIR / "nltk_data"

# Dossier de secours inscriptible (seul /tmp l'est sur Vercel)
NLTK_DIR_SECOURS = Path("/tmp/nltk_data")

# Indiquer à NLTK où chercher ses ressources
nltk.data.path.insert(0, str(NLTK_DIR))
nltk.data.path.append(str(NLTK_DIR_SECOURS))


def verifier_ressource(chemin, package):
    """
    Vérifie si une ressource NLTK existe.
    Si elle manque, elle est téléchargée automatiquement dans /tmp.
    """
    try:
        nltk.data.find(chemin)

    except LookupError:
        print(f"Téléchargement de la ressource NLTK : {package}")

        succes = nltk.download(
            package,
            download_dir=str(NLTK_DIR_SECOURS),
            quiet=True
        )

        if not succes:
            raise RuntimeError(
                f"Impossible de télécharger la ressource NLTK : {package}"
            )


# Ressources nécessaires au preprocessing
verifier_ressource("corpora/stopwords", "stopwords")
verifier_ressource("corpora/wordnet", "wordnet")

# ============================================================
# 2. STOPWORDS ET LEMMATISEUR
# ============================================================

# Stopwords anglais de NLTK
stop_words = set(stopwords.words("english"))

# Stopwords spécifiques identifiés dans le notebook 02
stop_words.update(["new", "year"])

# Initialisation du lemmatiseur
lemmatizer = WordNetLemmatizer()


# ============================================================
# 3. SUPPRESSION DU BRUIT
# ============================================================

def supprimer_bruit(texte):

    texte = re.sub(
        r" ?&?#(\d+);",
        lambda m: chr(int(m.group(1))),
        texte
    )

    texte = re.sub(r"&?quot;", '"', texte)

    texte = html.unescape(texte)

    # Suppression des publicités
    texte = re.sub(
        r"<p>\s*ADVERTISEMENT.*$",
        " ",
        texte
    )

    # Suppression des balises HTML
    texte = re.sub(r"<[^>]*>", " ", texte)

    # Suppression des liens
    texte = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        texte
    )

    # Suppression des liens boursiers Reuters
    texte = re.sub(
        r"\([^()]{0,15}Quote, Profile, Research\)",
        " ",
        texte
    )

    # Suppression des mentions (Reuters) Reuters -
    texte = re.sub(
        r"\(([^()]{1,40})\)\s*\1\s*-\s*",
        " ",
        texte
    )

    # Suppression des mentions NEW YORK (Reuters) -
    texte = re.sub(
        r"\([A-Za-z][A-Za-z. ]{1,30}\)\s+-\s+",
        " ",
        texte
    )

    # Suppression des antislashs
    texte = texte.replace("\\", " ")

    # Suppression des espaces multiples
    texte = re.sub(r"\s+", " ", texte)

    return texte.strip()


# ============================================================
# 4. LEMMATISATION
# ============================================================

@lru_cache(maxsize=None)
def lemmatiser(mot):

    return lemmatizer.lemmatize(mot)


# ============================================================
# 5. PREPROCESSING FINAL
# ============================================================

def preprocesser(texte):

    # Passage en minuscules
    texte = texte.lower()

    # Conservation uniquement des lettres
    texte = re.sub(r"[^a-z\s]", " ", texte)

    # Suppression des stopwords et mots trop courts
    mots = [
        m for m in texte.split()
        if m not in stop_words and len(m) > 2
    ]

    # Lemmatisation
    mots = [
        lemmatiser(m)
        for m in mots
    ]

    # Deuxième suppression des stopwords
    mots = [
        m for m in mots
        if m not in stop_words
    ]

    return " ".join(mots)
