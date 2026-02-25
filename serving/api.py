import os
import io
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel
import pdfplumber
import requests
import pickle
import subprocess
from sklearn.metrics.pairwise import cosine_similarity
import csv

# Membre 2 : API de service (Minimum Vital).
# Ce fichier contient uniquement les routes nécessaires pour tester le workflow n8n (Membre 4).

app = FastAPI(title="API Minimaliste pour Test n8n")

N8N_WEBHOOK_URL = "http://localhost:5678/webhook/explain-score"

class RequeteFeedback(BaseModel):
    user_feedback: bool
    cv_text: str = ""
    job_text: str = ""
    similarity_score: float = 0.0
    

    
# Variables globales pour stocker le modèle
modele_nlp = None
# Seuil réentrainement
SEUIL_K = 2


# Charger le modèle une seule fois
@app.on_event("startup")
def charger_modele():
    global modele_nlp
    chemin_modele = "artifacts/model.pickle"
    if os.path.exists(chemin_modele):
        with open(chemin_modele, "rb") as f:
            donnees = pickle.load(f)
            modele_nlp = donnees["model"] # On extrait juste l'objet SentenceTransformer
        print("Modèle NLP chargé avec succès au démarrage !")
    else:
        print("Attention : Fichier modèle introuvable dans les artifacts.")
        
        
    
@app.post("/predict")
async def prediction_simplifiee(fichier_cv: UploadFile = File(...), id_offre: str = Form(...)):
    try:
        # Lecture du PDF du CV
        contenu = await fichier_cv.read()
        with pdfplumber.open(io.BytesIO(contenu)) as pdf:
            texte_cv = pdf.pages[0].extract_text() or ""

        # Lecture du PDF de l'offre
        # L'API va chercher le fichier correspondant dans le dossier des offres
        chemin_offre = os.path.join("data", "jobs_corpus", id_offre)
        if not os.path.exists(chemin_offre):
            raise HTTPException(status_code=404, detail=f"L'offre {id_offre} est introuvable sur le serveur.")
            
        with pdfplumber.open(chemin_offre) as pdf_offre:
            texte_offre = pdf_offre.pages[0].extract_text() or ""

        # Calcul du vrai score avec le modèle
        if modele_nlp is None:
            raise HTTPException(status_code=500, detail="Le modèle n'est pas chargé en mémoire.")
            
        # On transforme les textes en vecteurs
        embedding_cv = modele_nlp.encode([texte_cv])
        embedding_offre = modele_nlp.encode([texte_offre])
        
        # On compare les vecteurs (le résultat est une matrice, on extrait la valeur [0][0])
        score_reel = float(cosine_similarity(embedding_cv, embedding_offre)[0][0])
        
        # Déclenchement de n8n avec les VRAIES données
        print(f"Appel du workflow n8n pour le CV {fichier_cv.filename}...")
        try:
            requests.post(N8N_WEBHOOK_URL, json={
                "cv_text": texte_cv,
                "job_text": texte_offre, # On envoie le vrai texte au LLM pour qu'il puisse justifier
                "score": score_reel      # On envoie le vrai score
            }, timeout=5)
        except Exception as e:
            print(f"Erreur lors de l'appel n8n : {e}")

        return {
            "similarity_score": score_reel,
            "message": "Prédiction effectuée avec succès. Workflow n8n déclenché.",
            "cv_text": texte_cv
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    

@app.post("/feedback")
def recevoir_feedback(donnees: RequeteFeedback):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dossier_data = os.path.join(base_dir, "data")
    chemin_prod = os.path.join(dossier_data, "prod_data.csv")
    
    os.makedirs(dossier_data, exist_ok=True)
    
    nouvelle_ligne = {
        'cv_text': donnees.cv_text,
        'job_text': donnees.job_text,
        'similarity_score': donnees.similarity_score,
        'user_feedback': donnees.user_feedback,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    try:
        fichier_existe = os.path.isfile(chemin_prod)
        with open(chemin_prod, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=nouvelle_ligne.keys())
            if not fichier_existe:
                writer.writeheader()
            writer.writerow(nouvelle_ligne)
        
        print(f"\n--- SUCCÈS : Donnée ajoutée dans {chemin_prod} ---")
        
    except Exception as e:
        print(f"ERREUR FATALE lors de l'écriture CSV : {e}")
        return {"statut": "erreur", "message": str(e)}

    # -- LOGIQUE DE SEUIL DE RÉENTRAÎNEMENT --
    try:
        df_prod = pd.read_csv(chemin_prod)
        nb_lignes = len(df_prod)
        print(f"Total des données en production : {nb_lignes}")
        
        # On déclenche uniquement si on est un multiple de SEUIL_K
        if nb_lignes > 0 and nb_lignes % SEUIL_K == 0:
            print(f"--- Seuil de {SEUIL_K} atteint. Lancement du réentraînement automatique ---")
            chemin_script = os.path.join(base_dir, "scripts", "retrain_model.py")
            subprocess.run(["python", chemin_script], check=True)
            
            # Rechargement à chaud du modèle après entraînement
            charger_modele()
            print("--- Réentraînement terminé et modèle mis à jour en production ! ---")
            
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution du script de réentraînement : {e}")
    except Exception as e:
        print(f"Erreur inattendue : {e}")

    return {"statut": "success", "message": "Feedback enregistré"}


import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)