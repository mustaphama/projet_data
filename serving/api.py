from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import pickle
import numpy as np
import io
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="CV Matching API")

# Modèle global
nlp_model = None

@app.on_event("startup")
def load_artifacts():
    global nlp_model
    try:
        # Charge le modèle (assure-toi que scripts/init_model.py a bien été lancé avant)
        with open("/artifacts/model.pickle", "rb") as f:
            nlp_model = pickle.load(f)
        print("✅ Modèle NLP chargé avec succès.")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")

def extract_text_from_pdf(file_content: bytes) -> str:
    """Fonction utilitaire pour extraire le texte d'un PDF binaire"""
    try:
        reader = PdfReader(io.BytesIO(file_content))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Impossible de lire le PDF: {str(e)}")

@app.post("/predict")
async def predict(
    cv_file: UploadFile = File(...),    # Le fichier PDF uploadé
    job_text: str = Form(...)           # Le texte de l'offre (envoyé comme champ de formulaire)
):
    if nlp_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # 1. Lecture et extraction du texte du PDF
    content = await cv_file.read()
    cv_text_extracted = extract_text_from_pdf(content)
    
    if not cv_text_extracted.strip():
        raise HTTPException(status_code=400, detail="Le PDF semble vide ou illisible.")

    # 2. Vectorisation (Embedding)
    # On encode le texte extrait du PDF et le texte de l'offre
    cv_vector = nlp_model.encode(cv_text_extracted).reshape(1, -1)
    job_vector = nlp_model.encode(job_text).reshape(1, -1)
    
    # 3. Calcul de la similarité
    score = cosine_similarity(cv_vector, job_vector)[0][0]
    
    return {
        "filename": cv_file.filename,
        "similarity_score": float(score),
        "match": bool(score > 0.5),
        # On renvoie un extrait du texte pour vérifier que l'extraction a marché (utile pour le debug)
        "extracted_text_snippet": cv_text_extracted[:200] + "..." 
    }