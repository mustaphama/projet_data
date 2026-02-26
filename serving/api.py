import os
import io
import glob
import pickle
import logging
import re
import csv
import subprocess
import requests
import uuid
import json
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import pdfplumber
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel

# Config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEUIL_K = 5

chemin_racine = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
chemin_racine = "/app"
DOSSER_ARTIFACTS = os.path.join(chemin_racine, 'artifacts')
CHEMIN_PICKLE_MODELE = os.path.join(DOSSER_ARTIFACTS, 'model.pickle')
CHEMIN_PENDING_PREDICTIONS = os.path.join(chemin_racine, "data", "pending_predictions.json")
DEFAULT_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

# --- AJOUT : Chargement de la PCA ---
CHEMIN_PCA = os.path.join(DOSSER_ARTIFACTS, 'pca.pickle')

def load_pca():
    if os.path.exists(CHEMIN_PCA):
        with open(CHEMIN_PCA, 'rb') as f:
            return pickle.load(f)
    return None

# URL du webhook n8n (ex: http://localhost:5678/webhook/explain-score)
N8N_WEBHOOK_URL = os.environ.get('N8N_WEBHOOK_URL', 'http://localhost:5678/webhook/explain-score')
N8N_TIMEOUT = 5  # sec pour l'appel n8n

app = FastAPI(title="Serving API - Real Matching")

def save_pending_prediction(prediction_id, data):
    try:
        current_data = {}
        os.makedirs(os.path.dirname(CHEMIN_PENDING_PREDICTIONS), exist_ok=True)
        if os.path.exists(CHEMIN_PENDING_PREDICTIONS):
            with open(CHEMIN_PENDING_PREDICTIONS, 'r', encoding='utf-8') as f:
                try:
                    current_data = json.load(f)
                except:
                    current_data = {}
        current_data[prediction_id] = data
        with open(CHEMIN_PENDING_PREDICTIONS, 'w', encoding='utf-8') as f:
            json.dump(current_data, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving pending prediction: {e}")

def get_pending_prediction(prediction_id):
    try:
        if os.path.exists(CHEMIN_PENDING_PREDICTIONS):
            with open(CHEMIN_PENDING_PREDICTIONS, 'r', encoding='utf-8') as f:
                current_data = json.load(f)
                return current_data.get(prediction_id)
    except Exception as e:
        logger.error(f"Error reading pending prediction: {e}")
    return None

class RequeteFeedback(BaseModel):
    user_feedback: bool
    prediction_id: Optional[str] = None
    cv_text: str = ""
    job_text: str = ""
    similarity_score: float = 0.0

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
PCA_MODELE = load_pca()
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
async def prediction_real(
    fichier_cv: UploadFile = File(...), 
    id_offre: Optional[str] = Form(None), 
    job_text: Optional[str] = Form(None),
    trigger_n8n: bool = Form(True)
):
    """
    Calcule le score réel entre le CV uploadé et une offre.
    - id_offre : si fourni, tu peux remplacer job_text par la lecture d'un fichier d'offres côté serveur
    - job_text : texte de l'offre envoyé dans le form
    - trigger_n8n : si True, lance le workflow d'explication n8n
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
            nom_fichier = id_offre if id_offre.endswith('.pdf') else f"{id_offre}.pdf"
            chemin_offre = os.path.join(chemin_racine, "data", "jobs_corpus", nom_fichier)
            if os.path.exists(chemin_offre):
                with open(chemin_offre, "rb") as f:
                    texte_offre_raw = extraire_texte_pdf_bytes(f.read())
            else:
                texte_offre_raw = f"Offre ID: {id_offre}"
        else:
            raise HTTPException(status_code=400, detail="Fournir job_text ou id_offre")

        texte_offre = nettoyer_texte(texte_offre_raw)
        score = compute_similarity(texte_cv, texte_offre)

        # Générer un ID unique pour cette prédiction
        pred_id = str(uuid.uuid4())
        save_pending_prediction(pred_id, {
            "cv_text": texte_cv_raw,
            "job_text": texte_offre_raw,
            "score": score
        })

        # Appel vers n8n seulement si demandé
        if trigger_n8n:
            payload = {
                "prediction_id": pred_id,
                "cv_text": texte_cv_raw,
                "job_text": texte_offre_raw,
                "score": score
            }
            try:
                resp = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=N8N_TIMEOUT)
                if resp.status_code >= 400:
                    logger.warning(f"n8n webhook retourné {resp.status_code}: {resp.text}")
            except Exception as e:
                logger.warning(f"Impossible d'appeler n8n webhook ({e})")

        return {
            "prediction_id": pred_id,
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
    

@app.get("/")
def health_check():
    return {"status": "L'API fonctionne parfaitement !"}


# Endpoint de feedback
@app.post("/feedback")
def recevoir_feedback(donnees: RequeteFeedback):
    global MODELE_NLP, PCA_MODELE

    chemin_prod = os.path.join(chemin_racine, "data", "prod_data.csv")
    os.makedirs(os.path.join(chemin_racine, "data"), exist_ok=True)
    
    cv_text = donnees.cv_text
    job_text = donnees.job_text
    score = donnees.similarity_score

    if donnees.prediction_id:
        pending = get_pending_prediction(donnees.prediction_id)
        if pending:
            cv_text = pending.get("cv_text", cv_text)
            job_text = pending.get("job_text", job_text)
            score = pending.get("score", score)
            logger.info(f"Données récupérées pour prediction_id: {donnees.prediction_id}")

    # --- AJOUT : Calcul des coordonnées PCA ---
    coord_cv = [0.0, 0.0]
    coord_job = [0.0, 0.0]
    if PCA_MODELE is not None:
        emb_cv = MODELE_NLP.encode([cv_text])
        emb_job = MODELE_NLP.encode([job_text])
        coord_cv = PCA_MODELE.transform(emb_cv)[0]
        coord_job = PCA_MODELE.transform(emb_job)[0]

    nouvelle_ligne = {
        'cv_text': cv_text,
        'job_text': job_text,
        'similarity_score': score,
        'cv_pca_1': coord_cv[0],
        'cv_pca_2': coord_cv[1],
        'job_pca_1': coord_job[0],
        'job_pca_2': coord_job[1],
        'user_feedback': 1 if donnees.user_feedback else 0, # Mettre 1 ou 0 pour Evidently
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    try:
        fichier_existe = os.path.isfile(chemin_prod)
        with open(chemin_prod, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=nouvelle_ligne.keys())
            if not fichier_existe:
                writer.writeheader()
            writer.writerow(nouvelle_ligne)
        logger.info(f"Feedback enregistré dans {chemin_prod}")
    except Exception as e:
        logger.exception("Erreur lors de l'écriture du feedback CSV")
        return {"statut": "erreur", "message": str(e)}

    # -- LOGIQUE DE SEUIL DE RÉENTRAÎNEMENT --
    try:
        df_prod = pd.read_csv(chemin_prod)
        nb_lignes = len(df_prod)
        logger.info(f"Total des données en production : {nb_lignes}")
        
        if nb_lignes > 0 and nb_lignes % SEUIL_K == 0:
            logger.info(f"Seuil de {SEUIL_K} atteint. Lancement du réentraînement automatique.")
            chemin_script = os.path.join(chemin_racine, "scripts", "retrain_model.py")
            subprocess.run(["python", chemin_script], check=True)
            
            # Rechargement à chaud
            
            MODELE_NLP = load_model()
            PCA_MODELE = load_pca() 
            logger.info("Réentraînement terminé et modèle mis à jour en production !")
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Erreur lors du script de réentraînement : {e}")
    except Exception as e:
        logger.error(f"Erreur inattendue : {e}")

    return {"statut": "success", "message": "Feedback enregistré"}


if __name__ == "__main__":
    try:
        import uvicorn
        port = int(os.environ.get("PORT", 8000))
        reload_flag = os.environ.get("UVICORN_RELOAD", "0") == "1"
        # On change l'hôte par défaut à 0.0.0.0
        host = os.environ.get("HOST", "0.0.0.0") 
        logger.info(f"Démarrage uvicorn sur {host}:{port} (reload requested={reload_flag})")
        
        if reload_flag:
            logger.warning("Reload demandé mais désactivé...")
        
        # Et SURTOUT on force 0.0.0.0 ici :
        uvicorn.run("api:app", host="0.0.0.0", port=8000) 
        
    except Exception as e:
        logger.error(f"Erreur au lancement: {e}")