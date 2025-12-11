from fastapi import FastAPI
import pickle
import pandas as pd
import uvicorn

app = FastAPI()

# Variables globales pour les artefacts 
model = None
scaler = None
pca = None

@app.on_event("startup")
def load_artifacts():
    global model, scaler, pca
    # Attention aux chemins : dans Docker, on montera le volume artifacts
    with open("/artifacts/model.pickle", "rb") as f:
        model = pickle.load(f)
    with open("/artifacts/scaler.pickle", "rb") as f:
        scaler = pickle.load(f)
    with open("/artifacts/pca.pickle", "rb") as f:
        pca = pickle.load(f)
    print("✅ Modèles chargés")

@app.post("/predict")
def predict(data: dict):
    # Logique à implémenter plus tard [cite: 71]
    return {"message": "API is running", "prediction": "To be implemented"}