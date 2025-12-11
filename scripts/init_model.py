import pandas as pd
import numpy as np
import pickle
import os
import glob
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Chemins
pdfs_dir = os.path.join(project_root, 'data', 'resumes_corpus')
ref_data_path = os.path.join(project_root, 'data', 'ref_data.csv')
artifacts_path = os.path.join(project_root, 'artifacts')

# Liste d'offres d'emploi simulées pour créer des paires de comparaison
# (Idéalement, tu devrais aussi avoir un dossier de PDF d'offres, mais on simplifie ici)
SAMPLE_JOBS = [
    "Data Scientist senior avec expertise en NLP, Python et Docker. Expérience en déploiement de modèles requise.",
    "Développeur Fullstack React/NodeJS pour concevoir des interfaces web modernes.",
    "Chef de projet IT, certification PMP, gestion d'équipe agile et budget.",
    "Stage assistant RH, gestion de la paie et recrutement.",
    "Ingénieur DevOps, maîtrise de Kubernetes, AWS et CI/CD pipelines."
]

def extract_text_from_pdf(pdf_path):
    """Lit un fichier PDF et retourne son texte."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t: text += t + "\n"
        return text
    except Exception as e:
        print(f"⚠️ Erreur de lecture sur {pdf_path}: {e}")
        return ""

# --- 1. CHARGEMENT DU MODÈLE ---
# On utilise le modèle multilingue performant que nous avons choisi
model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
print(f"⏳ Chargement du modèle {model_name}...")
model = SentenceTransformer(model_name)

# --- 2. TRAITEMENT DES PDFS ---
pdf_files = glob.glob(os.path.join(pdfs_dir, "*.pdf"))
if not pdf_files:
    print(f"❌ AUCUN PDF TROUVÉ DANS : {pdfs_dir}")
    print(">> Veuillez créer le dossier et y mettre vos CVs.")
    exit(1)

print(f"📂 {len(pdf_files)} CVs trouvés. Génération des embeddings...")

rows = []

for pdf_path in pdf_files:
    filename = os.path.basename(pdf_path)
    print(f"   -> Traitement de : {filename}")
    
    cv_text = extract_text_from_pdf(pdf_path)
    if len(cv_text) < 50:
        print(f"      ⚠️ Ignoré (Texte trop court ou illisible)")
        continue

    # Vectorisation du CV
    cv_emb = model.encode(cv_text)

    # Pour construire ref_data.csv, on compare ce CV à chacune des offres exemples
    # Cela permet de créer des cas de "Match" et de "Non Match" pour tes stats
    for job_desc in SAMPLE_JOBS:
        job_emb = model.encode(job_desc)
        
        # Calcul du score (pour la target simulée)
        score = cosine_similarity([cv_emb], [job_emb])[0][0]
        
        # On considère arbitrairement qu'un score > 0.4 est un "Match" (True)
        is_match = bool(score > 0.4)
        
        # Création de la ligne pour le CSV
        # On stocke quelques dimensions du vecteur différence + la target + la prediction
        # (C'est ce format que Evidently utilisera pour surveiller le drift)
        diff_vector = np.abs(cv_emb - job_emb)
        
        row = {f'emb_{i}': val for i, val in enumerate(diff_vector[:10])} # 10 premières features
        row['target'] = is_match     # La "vérité terrain" (ici simulée par le score)
        row['prediction'] = is_match # Ce que notre modèle prédit (ici idem car pas de bruit)
        row['similarity'] = score    # Utile pour l'analyse
        rows.append(row)

# --- 3. SAUVEGARDE DU CSV ---
df = pd.DataFrame(rows)
df.to_csv(ref_data_path, index=False)
print(f"✅ 'data/ref_data.csv' généré avec {len(df)} lignes.")

# --- 4. SAUVEGARDE DU MODÈLE (ARTIFACTS) ---
model_pickle_path = os.path.join(artifacts_path, 'model.pickle')
with open(model_pickle_path, 'wb') as f:
    pickle.dump(model, f)

print(f"✅ Modèle sauvegardé dans : {model_pickle_path}")