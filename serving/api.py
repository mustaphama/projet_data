import os
import io
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel
import pdfplumber
import requests

# Membre 2 : API de service (Minimum Vital).
# Ce fichier contient uniquement les routes nécessaires pour tester le workflow n8n (Membre 4).

app = FastAPI(title="API Minimaliste pour Test n8n")

N8N_WEBHOOK_URL = "http://localhost:5678/webhook/explain-score"

class RequeteFeedback(BaseModel):
    user_feedback: bool
    cv_text: str = ""
    job_text: str = ""
    similarity_score: float = 0.0

@app.post("/predict")
async def prediction_simplifiee(fichier_cv: UploadFile = File(...), id_offre: str = Form(...)):
    """
    Simule une prédiction et déclenche le workflow n8n pour explication.
    TO-DO : Intégrer le vrai modèle SentenceTransformer (Membre 2).
    """
    try:
        # Lecture minimaliste du PDF
        contenu = await fichier_cv.read()
        with pdfplumber.open(io.BytesIO(contenu)) as pdf:
            texte_cv = pdf.pages[0].extract_text() or "Texte CV vide"

        # Simulation de score
        score_simule = 0.75
        
        # Déclenchement de n8n (Membre 4)
        print(f"Appel du workflow n8n pour le CV {fichier_cv.filename}...")
        try:
            requests.post(N8N_WEBHOOK_URL, json={
                "cv_text": texte_cv,
                "job_text": f"Offre ID: {id_offre}",
                "score": score_simule
            }, timeout=5)
        except Exception as e:
            print(f"Erreur lors de l'appel n8n (vérifiez si n8n est lancé) : {e}")

        return {
            "similarity_score": score_simule,
            "message": "Prédiction simulée. Workflow n8n déclenché.",
            "cv_text": texte_cv
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
def recevoir_feedback(donnees: RequeteFeedback):
    """
    Endpoint appelé par n8n à la fin du workflow de validation humaine.
    """
    print(f"Feedback reçu de n8n : {'Validé' if donnees.user_feedback else 'Rejeté'}")
    
    # TO-DO : Sauvegarder dans prod_data.csv et lancer retrain_model (Membre 2).
    return {"statut": "success", "info": "Le feedback a bien été reçu par l'API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
