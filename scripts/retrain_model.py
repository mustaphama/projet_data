import pandas as pd
import pickle
import os
import logging
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

# Membre 1 (Albin) : Script de réentraînement du modèle.
# Fusionne les données initiales avec les feedbacks de production.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

chemin_racine = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
chemin_csv_reference = os.path.join(chemin_racine, 'data', 'ref_data.csv')
chemin_csv_production = os.path.join(chemin_racine, 'data', 'prod_data.csv')
dossier_artifacts = os.path.join(chemin_racine, 'artifacts')
chemin_pickle_modele = os.path.join(dossier_artifacts, 'model.pickle')
chemin_pickle_pca = os.path.join(dossier_artifacts, 'pca.pickle')

def retrain_model():
    # procédure de mise à jour du modèle en tâche de fond
    logger.info("Début de la phase de réentraînement...")
    
    if not os.path.exists(chemin_csv_reference):
        logger.error(f"Erreur : Fichier de référence absent à l'emplacement {chemin_csv_reference}")
        return False
    
    donnees_globales = pd.read_csv(chemin_csv_reference)
    
    if os.path.exists(chemin_csv_production):
        df_production = pd.read_csv(chemin_csv_production)
        # filtrage des feedbacks validés par l'utilisateur
        if 'user_feedback' in df_production.columns:
            feedbacks_positifs = df_production[df_production["user_feedback"] == True]
            logger.info(f"Intégration de {len(feedbacks_positifs)} nouveaux exemples validés.")
            donnees_globales = pd.concat([donnees_globales, feedbacks_positifs], ignore_index=True)
    
    # Chargement du modèle SentenceTransformer
    nom_modele = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    modele_nlp = SentenceTransformer(nom_modele)
    
    logger.info("Calcul des nouveaux embeddings pour le dataset fusionné...")
    embeddings_cv = modele_nlp.encode(donnees_globales["cv_text"].tolist())
    embeddings_offres = modele_nlp.encode(donnees_globales["job_text"].tolist())
    
    # Calcul des scores de similarité mis à jour
    scores = [float(cosine_similarity([c], [j])[0][0]) for c, j in zip(embeddings_cv, embeddings_offres)]
    donnees_globales["similarity_score"] = scores
    
    score_moyen = np.mean(scores)
    logger.info(f"Score de similarité moyen après mise à jour : {score_moyen:.4f}")
    
    # Calcul des métriques de performance globales
    if not donnees_globales["user_feedback"].isna().all():
        seuil_validation = 0.4
        valeurs_reelles = donnees_globales["user_feedback"].fillna(False).astype(bool).tolist()
        predictions = [s >= seuil_validation for s in scores]
        
        tp = sum((r and p) for r, p in zip(valeurs_reelles, predictions))
        fp = sum((not r and p) for r, p in zip(valeurs_reelles, predictions))
        fn = sum((r and not p) for r, p in zip(valeurs_reelles, predictions))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        rappel = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        logger.info(f"Métriques de stabilité au seuil {seuil_validation} : Précision={precision:.2f}, Rappel={rappel:.2f}")

    # Mise à jour de la projection PCA
    logger.info("Recalcul de la projection PCA sur les nouvelles données...")
    embeddings_combines = np.vstack([embeddings_cv, embeddings_offres])
    pca_nouveau = PCA(n_components=2)
    pca_nouveau.fit(embeddings_combines)
    
    # Transformation des coordonnées
    coordonnees_cv = pca_nouveau.transform(embeddings_cv)
    coordonnees_offres = pca_nouveau.transform(embeddings_offres)
    donnees_globales['cv_pca_1'], donnees_globales['cv_pca_2'] = coordonnees_cv[:, 0], coordonnees_cv[:, 1]
    donnees_globales['job_pca_1'], donnees_globales['job_pca_2'] = coordonnees_offres[:, 0], coordonnees_offres[:, 1]
    
    # Sauvegarde des artifacts
    if not os.path.exists(dossier_artifacts):
        os.makedirs(dossier_artifacts)
    
    with open(chemin_pickle_modele, 'wb') as f:
        pickle.dump({
            "model": modele_nlp,
            "date_mise_a_jour": datetime.now().isoformat(),
            "score_moyen": score_moyen
        }, f)
        
    with open(chemin_pickle_pca, 'wb') as f:
        pickle.dump(pca_nouveau, f)
        
    # Mise à jour du fichier de référence pour les itérations futures
    donnees_globales.to_csv(chemin_csv_reference, index=False)
    
    logger.info("Réentraînement terminé avec succès.")
    return True

if __name__ == "__main__":
    retrain_model()
