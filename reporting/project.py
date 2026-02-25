import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.ui.workspace import Workspace
from evidently import ColumnMapping # Import nécessaire pour la nouvelle API

# Chemins vers les données partagées via le volume Docker
REF_PATH = "/app/data/ref_data.csv"
PROD_PATH = "/app/data/prod_data.csv"
WORKSPACE_PATH = "workspace"

def preparer_donnees(df):
    """
    Adapte le dataset pour Evidently.
    Evidently a besoin d'une colonne 'target' et 'prediction'[cite: 169].
    """
    df = df.copy()
    
    # Simulation d'une prédiction binaire basée sur le score de similarité
    if 'similarity_score' in df.columns:
        df['prediction'] = (df['similarity_score'] >= 0.4).astype(int)
    else:
        df['prediction'] = 0

    # 'target' est la réponse réelle (feedback utilisateur) [cite: 161]
    if 'user_feedback' in df.columns and not df['user_feedback'].isna().all():
        df['target'] = df['user_feedback'].fillna(df['prediction']).astype(int)
    else:
        # Si pas de feedback (cas de ref_data), on aligne sur la prédiction
        df['target'] = df['prediction']
        
    return df

def generer_rapport_evidently():
    print("Initialisation du Workspace Evidently...")
    if not os.path.exists(WORKSPACE_PATH):
        os.makedirs(WORKSPACE_PATH)
    ws = Workspace.create(WORKSPACE_PATH)
    
    # Création ou récupération du projet [cite: 170]
    project_name = "Projet Data - Monitoring"
    projects = ws.search_project(project_name)
    if not projects:
        project = ws.create_project(project_name)
        project.description = "Monitoring du Data Drift et des performances [cite: 150]"
        project.save()
    else:
        project = projects[0]

    print("Chargement des données...")
    if not os.path.exists(REF_PATH):
        print(f"Erreur : Fichier de référence introuvable ({REF_PATH}) [cite: 153]")
        return

    ref_data = pd.read_csv(REF_PATH)
    ref_data = preparer_donnees(ref_data)

    if os.path.exists(PROD_PATH):
        prod_data = pd.read_csv(PROD_PATH)
        prod_data = preparer_donnees(prod_data)
        print(f"Données de production chargées : {len(prod_data)} lignes [cite: 154]")
    else:
        print("Aucune donnée de production trouvée. Utilisation d'un échantillon de référence.")
        prod_data = ref_data.sample(frac=0.5, replace=True) 

    # Configuration du ColumnMapping pour les colonnes PCA [cite: 58, 169]
    # On liste les colonnes qui servent de base au calcul du drift [cite: 150]
    # Assurez-vous que les types sont cohérents pour éviter les WARNINGS
    ref_data['timestamp'] = pd.to_datetime(ref_data['timestamp'])
    prod_data['timestamp'] = pd.to_datetime(prod_data['timestamp'])
    
    # Correction de l'erreur : on ne garde que les colonnes présentes dans les DEUX datasets
    colonnes_numeriques = ['similarity_score', 'cv_pca_1', 'cv_pca_2', 'job_pca_1', 'job_pca_2']
    features_communes = [
        col for col in colonnes_numeriques 
        if col in ref_data.columns and col in prod_data.columns
    ]

    column_mapping = ColumnMapping()
    column_mapping.target = 'target'
    column_mapping.prediction = 'prediction'
    column_mapping.numerical_features = features_communes # On utilise la liste filtrée

    print(f"Colonnes surveillées pour le drift : {features_communes}")

    report = Report(metrics=[
        DataDriftPreset(),
        ClassificationPreset()
    ])

    # Exécution du rapport avec le mapping de colonnes
    report.run(reference_data=ref_data, current_data=prod_data, column_mapping=column_mapping)

    # Ajout au dashboard
    ws.add_report(project.id, report)
    print("Rapport généré avec succès !")

if __name__ == "__main__":
    generer_rapport_evidently()