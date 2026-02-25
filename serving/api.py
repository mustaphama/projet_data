import os
import io
import glob
import pickle
import logging
import re
import requests
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

chemin_racine = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOSSER_ARTIFACTS = os.path.join(chemin_racine, 'artifacts')
CHEMIN_PICKLE_MODELE = os.path.join(DOSSER_ARTIFACTS, 'model.pickle')
DEFAULT_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

# URL du webhook n8n (ex: http://localhost:5678/webhook/explain-score)
N8N_WEBHOOK_URL = os.environ.get('N8N_WEBHOOK_URL', 'http://localhost:5678/webhook/explain-score')
N8N_TIMEOUT = 5  # sec pour l'appel n8n

app = FastAPI(title="Serving API - Real Matching")

# mots vides simple (tu peux compléter)
MOTS_VIDES_FR = {"le", "la", "les", "un", "une", "des", "et", "en", "pour", "par", "dans", "sur", "avec"}
MOTS_VIDES_EN = {"the", "and", "for", "with", "from", "this", "that", "these", "those"}
TOUS_MOTS_VIDES = MOTS_VIDES_FR.union(MOTS_VIDES_EN)

# Chargement du modèle (global, une seule fois)
def load_model():
    # Try load pickled model if present (pickled dict with "model" or direct SentenceTransformer)
    if os.path.exists(CHEMIN_PICKLE_MODELE):
        try:
            with open(CHEMIN_PICKLE_MODELE, 'rb') as f:
                obj = pickle.load(f)
                # Cas où on a dict {"model": modele_nlp, ...}
                if isinstance(obj, dict) and "model" in obj:
                    logger.info("Chargement du modèle depuis model.pickle (clé 'model').")
                    return obj["model"]
                # Cas où pickle est directement le modèle
                logger.info("Chargement du modèle picklé depuis model.pickle.")
                return obj
        except Exception as e:
            logger.warning(f"Impossible de charger model.pickle ({e}), on retombe sur SentenceTransformer('{DEFAULT_MODEL_NAME}').")
    # Fallback : créer une instance
    logger.info(f"Initialisation SentenceTransformer('{DEFAULT_MODEL_NAME}')")
    return SentenceTransformer(DEFAULT_MODEL_NAME)

MODELE_NLP = load_model()

# utilitaires extraction/cleanup (repris et simplifiés de scripts/init_model.py)
def extraire_texte_pdf_bytes(contenu_bytes: bytes) -> str:
    try:
        textes = []
        with pdfplumber.open(io.BytesIO(contenu_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text(layout=True)
                if t:
                    textes.append(t)
        return "\n".join(textes)
    except Exception as e:
        logger.exception("Erreur extraction PDF")
        return ""

def nettoyer_texte(texte: str) -> str:
    if not texte:
        return ""
    texte = texte.replace('\n', ' ').replace('\r', ' ')
    texte = re.sub(r'\S+@\S+', '', texte)
    texte = re.sub(r'http\S+', '', texte)
    texte = re.sub(r'[^a-zA-Z0-9àâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ\s]', ' ', texte)
    texte = re.sub(r'\s+', ' ', texte).lower().strip()
    mots = [m for m in texte.split() if m not in TOUS_MOTS_VIDES and len(m) > 2]
    return " ".join(mots)

def compute_similarity(text_a: str, text_b: str) -> float:
    # encode retourne np.array ; on force convert_to_numpy pour cohérence
    emb_a = MODELE_NLP.encode([text_a], convert_to_numpy=True)
    emb_b = MODELE_NLP.encode([text_b], convert_to_numpy=True)
    # cosine_similarity attend matrices 2D
    score = float(cosine_similarity(emb_a, emb_b)[0][0])
    return score

@app.post("/predict")
async def prediction_real(fichier_cv: UploadFile = File(...), id_offre: Optional[str] = Form(None), job_text: Optional[str] = Form(None)):
    """
    Calcule le score réel entre le CV uploadé et une offre.
    - id_offre : si fourni, tu peux remplacer job_text par la lecture d'un fichier d'offres côté serveur (extension possible)
    - job_text : texte de l'offre envoyé dans le form
    """
    try:
        contenu = await fichier_cv.read()
        texte_cv_raw = extraire_texte_pdf_bytes(contenu)
        texte_cv = nettoyer_texte(texte_cv_raw)
        if not texte_cv or len(texte_cv) < 20:
            raise HTTPException(status_code=400, detail="Texte CV trop court ou non extrait correctement")
        
        # job_text prend la priorité si fourni
        if job_text:
            texte_offre_raw = job_text
        elif id_offre:
            # Optionnel : recherche d'un fichier offre local par id (ex: 'offre_1.pdf')
            nom_fichier = id_offre if id_offre.endswith('.pdf') else f"{id_offre}.pdf"
            chemin_offre = os.path.join(chemin_racine, "data", "jobs_corpus", nom_fichier)
            if os.path.exists(chemin_offre):
                with open(chemin_offre, "rb") as f:
                    texte_offre_raw = extraire_texte_pdf_bytes(f.read())
            else:
                texte_offre_raw = f"Offre ID: {id_offre}"  # fallback minimal
        else:
            raise HTTPException(status_code=400, detail="Fournir job_text ou id_offre")

        texte_offre = nettoyer_texte(texte_offre_raw)
        score = compute_similarity(texte_cv, texte_offre)

        # Appel vers n8n pour demander l'explication (le workflow n8n fait l'explication)
        payload = {
            "cv_text": texte_cv_raw,     # envoi le texte brut si tu veux que n8n/LLM voit plus de contexte
            "job_text": texte_offre_raw,
            "score": score
        }
        try:
            resp = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=N8N_TIMEOUT)
            if resp.status_code >= 400:
                logger.warning(f"n8n webhook retourné {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.warning(f"Impossible d'appeler n8n webhook ({e}) - l'explication via n8n peut ne pas être déclenchée")

        return {
            "similarity_score": score,
            "message": "Score calculé avec embeddings. Explication déclenchée via n8n (si reachable).",
            "cv_text_excerpt": texte_cv[:1000],
            "job_text_excerpt": texte_offre[:1000]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Erreur dans /predict")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint de feedback conservé (n8n peut poster ici après validation humaine)
from pydantic import BaseModel
class RequeteFeedback(BaseModel):
    user_feedback: bool
    cv_text: str = ""
    job_text: str = ""
    similarity_score: float = 0.0

@app.post("/feedback")
def recevoir_feedback(donnees: RequeteFeedback):
    # Enregistrement minimal: append to data/prod_data.csv
    try:
        import pandas as pd
        chemin_prod = os.path.join(chemin_racine, "data", "prod_data.csv")
        row = {
            "cv_text": donnees.cv_text,
            "job_text": donnees.job_text,
            "similarity_score": donnees.similarity_score,
            "user_feedback": donnees.user_feedback,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        if os.path.exists(chemin_prod):
            df = pd.read_csv(chemin_prod)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(chemin_prod, index=False)
        logger.info("Feedback reçu et enregistré.")
        return {"statut": "success"}
    except Exception as e:
        logger.exception("Erreur lors de l'enregistrement du feedback")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Démarre un serveur Uvicorn quand ce module est exécuté directement.
    # Contrôles via variables d'environnement : PORT et UVICORN_RELOAD (valeur '1' pour activer reload)
    try:
        import uvicorn
        port = int(os.environ.get("PORT", 8000))
        reload_flag = os.environ.get("UVICORN_RELOAD", "0") == "1"
        host = os.environ.get("HOST", "127.0.0.1")
        logger.info(f"Démarrage uvicorn sur {host}:{port} (reload requested={reload_flag})")
        # Si on exécute le script directement, il est plus fiable de passer l'objet app à uvicorn.
        # Le reloader d'uvicorn nécessite d'importer le module par nom (string). Pour éviter
        # le ModuleNotFoundError lorsque l'on exécute `python serving/api.py`, on désactive
        # automatique le reload ici et on recommande d'utiliser la commande uvicorn CLI
        # si le reloader est nécessaire.
        if reload_flag:
            logger.warning("Reload demandé mais désactivé lorsque le script est exécuté directement.\n"
                           "Pour utiliser reload, lancez: uvicorn serving.api:app --reload --port {port}")
        uvicorn.run(app, host=host, port=port, reload=False)
    except Exception as e:
        logger.exception("Erreur lors du démarrage du serveur Uvicorn: %s", e)