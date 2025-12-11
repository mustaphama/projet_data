from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel
import pickle
import pandas as pd
import os
import io
from datetime import datetime
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="CV Matching API")

# --- CONFIGURATION ---
ARTIFACTS_DIR = "/artifacts"
DATA_DIR = "/data"
PROD_DATA_PATH = os.path.join(DATA_DIR, "prod_data.csv")
RETRAIN_THRESHOLD_K = 5  # Pour le test, on réentraîne tous les 5 feedbacks (PDF: "seuil k")

# Variables globales
nlp_model = None

# Modèle de données pour le feedback
class FeedbackRequest(BaseModel):
    cv_text: str
    job_text: str
    similarity_score: float
    user_feedback: bool # True si l'utilisateur valide le match, False sinon

@app.on_event("startup")
def load_artifacts():
    global nlp_model
    try:
        with open(f"{ARTIFACTS_DIR}/model.pickle", "rb") as f:
            nlp_model = pickle.load(f)
        print("✅ Modèle NLP chargé.")
    except Exception as e:
        print(f"❌ Erreur chargement modèle: {e}")

def extract_text_from_pdf(file_content: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_content))
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail="PDF illisible")

def retrain_model_task():
    """Tâche de fond simulée pour le réentraînement (Trigger)"""
    print("🔄 TRIGGER ACTIVÉ : Lancement du réentraînement sur ref + prod...")
    # ICI : Tu devras appeler ton script de réentraînement (partie suivante)
    # Pour l'instant, on met juste un log
    pass

@app.post("/predict")
async def predict(cv_file: UploadFile = File(...), job_text: str = Form(...)):
    if nlp_model is None: raise HTTPException(500, "Modèle non chargé")
    
    content = await cv_file.read()
    cv_text = extract_text_from_pdf(content)
    
    # Vectorisation
    cv_vec = nlp_model.encode(cv_text).reshape(1, -1)
    job_vec = nlp_model.encode(job_text).reshape(1, -1)
    score = float(cosine_similarity(cv_vec, job_vec)[0][0])
    
    return {
        "similarity_score": score,
        "match": bool(score > 0.4), # Seuil arbitraire
        "cv_text_extract": cv_text[:500] # On renvoie le texte pour qu'il soit renvoyé au feedback plus tard
    }

@app.post("/feedback")
def submit_feedback(feedback: FeedbackRequest, background_tasks: BackgroundTasks):
    """Reçoit la validation utilisateur et trigger le réentraînement si besoin [cite: 190, 191]"""
    
    # 1. Sauvegarde dans prod_data.csv
    new_row = {
        'cv_text': feedback.cv_text,
        'job_text': feedback.job_text,
        'similarity_score': feedback.similarity_score,
        'user_feedback': feedback.user_feedback,
        'timestamp': datetime.now().isoformat()
    }
    
    # On utilise le mode 'a' (append) pour ajouter sans tout relire
    df = pd.DataFrame([new_row])
    # Si le fichier n'existe pas, on met l'header, sinon non
    header = not os.path.exists(PROD_DATA_PATH)
    df.to_csv(PROD_DATA_PATH, mode='a', header=header, index=False)
    
    # 2. Vérification du Trigger (Seuil k) [cite: 192]
    try:
        # On compte le nombre de lignes (c'est un peu lourd de tout lire, mais simple pour le TP)
        current_df = pd.read_csv(PROD_DATA_PATH)
        count = len(current_df)
        print(f"📊 Feedback reçu. Total données prod: {count}")
        
        if count % RETRAIN_THRESHOLD_K == 0:
            # Trigger atteint ! On lance le réentraînement en tâche de fond pour ne pas bloquer l'API
            background_tasks.add_task(retrain_model_task)
            return {"message": "Feedback enregistré. 🚀 Réentraînement déclenché !"}
            
    except Exception as e:
        print(f"Erreur lecture prod_data: {e}")
        
    return {"message": "Feedback enregistré"}