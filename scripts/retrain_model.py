import pandas as pd
import pickle
import os
import logging
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
REF_DATA_PATH = os.path.join(project_root, 'data', 'ref_data.csv')
PROD_DATA_PATH = os.path.join(project_root, 'data', 'prod_data.csv')
MODEL_PATH = os.path.join(project_root, 'artifacts', 'model.pickle')

def retrain_model():
    """
    Réentraîne le modèle en fusionnant les données de référence
    et les feedbacks de production avec user_feedback == True
    """
    logger.info("🔄 Démarrage du réentraînement incrémental...")
    
    # 1. Charger les données de référence
    if not os.path.exists(REF_DATA_PATH):
        logger.error(f"❌ {REF_DATA_PATH} introuvable")
        return False
    
    ref_data = pd.read_csv(REF_DATA_PATH)
    logger.info(f"📂 Données de référence: {len(ref_data)} lignes")
    
    # 2. Charger les feedbacks de production
    if not os.path.exists(PROD_DATA_PATH):
        logger.warning(f"⚠️  {PROD_DATA_PATH} introuvable, utilisation de ref_data uniquement")
        all_data = ref_data.copy()
    else:
        prod_data = pd.read_csv(PROD_DATA_PATH)
        logger.info(f"📂 Données de production: {len(prod_data)} lignes")
        
        # Filtrer les feedbacks positifs (user_feedback == True)
        positive_feedback = prod_data[prod_data["user_feedback"] == True]
        logger.info(f"✅ Feedbacks positifs: {len(positive_feedback)}")
        
        # Fusionner ref_data + feedbacks positifs
        all_data = pd.concat([ref_data, positive_feedback], ignore_index=True)
    
    logger.info(f"📊 Données totales après fusion: {len(all_data)} lignes")
    
    # 3. Charger le modèle
    logger.info(f"📥 Chargement du modèle {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    # 4. Régénérer les embeddings pour tous les CVs
    logger.info("🧠 Régénération des embeddings...")
    cv_texts = all_data["cv_text"].tolist()
    job_texts = all_data["job_text"].tolist()
    
    cv_embeddings = model.encode(cv_texts, show_progress_bar=True)
    job_embeddings = model.encode(job_texts, show_progress_bar=True)
    
    logger.info(f"✅ Embeddings générés: CV {cv_embeddings.shape}, Jobs {job_embeddings.shape}")
    
    # 5. Recalculer les scores de similarité
    similarity_scores = []
    for i in range(len(cv_embeddings)):
        score = float(cosine_similarity([cv_embeddings[i]], [job_embeddings[i]])[0][0])
        similarity_scores.append(score)
    
    all_data["similarity_score"] = similarity_scores
    
    # 6. Sauvegarder le nouveau modèle et les données
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({
            "model": model,
            "data": all_data,
            "retrain_date": datetime.now().isoformat()
        }, f)
    
    logger.info(f"💾 Modèle réentraîné sauvegardé: {MODEL_PATH}")
    logger.info("🎉 Réentraînement terminé avec succès!")
    
    return True

if __name__ == "__main__":
    success = retrain_model()
    exit(0 if success else 1)
