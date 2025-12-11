import pandas as pd
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
jobs_dir = os.path.join(project_root, 'data', 'jobs_corpus')
ref_data_path = os.path.join(project_root, 'data', 'ref_data.csv')
prod_data_path = os.path.join(project_root, 'data', 'prod_data.csv')
artifacts_path = os.path.join(project_root, 'artifacts')

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

def clean_text(text):
    """Normalise les espaces dans le texte"""
    import re
    # Supprimer les espaces entre les caractères individuels (A L B I N -> ALBIN)
    text = re.sub(r'(?<=[A-Za-zÀ-ÿ])\s(?=[A-Za-zÀ-ÿ])', '', text)
    # Normaliser les espaces multiples
    text = ' '.join(text.split())
    return text.strip()

# --- 1. CHARGEMENT DU MODÈLE ---
model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
print(f"⏳ Chargement du modèle {model_name}...")
model = SentenceTransformer(model_name)

# --- 2. CHARGEMENT DES OFFRES ---
job_files = glob.glob(os.path.join(jobs_dir, "*.pdf"))
if not job_files:
    print(f"❌ AUCUNE OFFRE TROUVÉE DANS : {jobs_dir}")
    exit(1)

job_texts = []
for job_path in job_files:
    job_text = extract_text_from_pdf(job_path)
    job_text = clean_text(job_text)
    if len(job_text) > 50:
        job_texts.append(job_text)
        print(f"   ✅ Offre chargée: {os.path.basename(job_path)}")

print(f"📂 {len(job_texts)} offres chargées.")

# --- 3. TRAITEMENT DES CVSS ---
pdf_files = glob.glob(os.path.join(pdfs_dir, "*.pdf"))
if not pdf_files:
    print(f"❌ AUCUN PDF TROUVÉ DANS : {pdfs_dir}")
    exit(1)

print(f"📂 {len(pdf_files)} CVs trouvés. Génération des embeddings...")

rows = []

for pdf_path in pdf_files:
    filename = os.path.basename(pdf_path)
    print(f"   -> Traitement de : {filename}")
    
    cv_text = extract_text_from_pdf(pdf_path)
    cv_text = clean_text(cv_text)
    if len(cv_text) < 50:
        print(f"      ⚠️ Ignoré (Texte trop court ou illisible)")
        continue

    # Pour construire ref_data.csv, on compare ce CV à chacune des offres
    for job_text in job_texts:
        # Calcul du score de similarité
        score = float(cosine_similarity(
            [model.encode(cv_text)], 
            [model.encode(job_text)]
        )[0][0])
        
        row = {
            'cv_text': cv_text,
            'job_text': job_text,
            'similarity_score': score,
            'user_feedback': None,
            'timestamp': None
        }
        rows.append(row)

# --- 3. SAUVEGARDE DU CSV FUSIONNÉ ---
df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(ref_data_path), exist_ok=True)
df.to_csv(ref_data_path, index=False)
print(f"✅ 'data/ref_data.csv' généré avec {len(df)} lignes.")

# --- 4. SAUVEGARDE DU MODÈLE (ARTIFACTS) ---
model_pickle_path = os.path.join(artifacts_path, 'model.pickle')
with open(model_pickle_path, 'wb') as f:
    pickle.dump(model, f)

print(f"✅ Modèle sauvegardé dans : {model_pickle_path}")