import pandas as pd
import pickle
import os
import glob
import pdfplumber
import re
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

# Membre 1 (Albin) : Script d'initialisation du modèle et du dataset de référence.
# Ce script prépare les données, calcule les embeddings et applique une PCA.

# On désactive les logs inutiles de pdfminer
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# Définition des dossiers de travail
chemin_racine = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dossier_cv = os.path.join(chemin_racine, 'data', 'resumes_corpus')
dossier_offres = os.path.join(chemin_racine, 'data', 'jobs_corpus')
chemin_csv_reference = os.path.join(chemin_racine, 'data', 'ref_data.csv')
dossier_artifacts = os.path.join(chemin_racine, 'artifacts')

if not os.path.exists(dossier_artifacts):
    os.makedirs(dossier_artifacts)

# Liste des mots vides pour le nettoyage du texte
mots_vides_fr = {"le", "la", "les", "un", "une", "des", "et", "en", "pour", "par", "dans", "sur", "avec"}
mots_vides_en = {"the", "and", "for", "with", "from", "this", "that", "these", "those"}
tous_les_mots_vides = mots_vides_fr.union(mots_vides_en)

def extraire_texte_pdf(chemin_fichier):
    # extraction du texte avec pdfplumber
    try:
        contenu = []
        with pdfplumber.open(chemin_fichier) as pdf:
            for page in pdf.pages:
                texte_page = page.extract_text(layout=True)
                if texte_page:
                    contenu.append(texte_page)
        return "\n".join(contenu)
    except Exception as e:
        print(f"Erreur lors de la lecture de {os.path.basename(chemin_fichier)} : {e}")
        return ""

def nettoyer_texte(texte):
    # nettoyage standard : minuscules, suppression urls et emails
    if not texte:
        return ""
    
    texte = texte.replace('\n', ' ').replace('\r', ' ')
    texte = re.sub(r'\S+@\S+', '', texte)
    texte = re.sub(r'http\S+', '', texte)
    
    # on conserve les caractères alphanumériques et les accents
    texte = re.sub(r'[^a-zA-Z0-9àâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ\s]', ' ', texte)
    texte = re.sub(r'\s+', ' ', texte).lower().strip()
    
    # suppression des mots vides
    mots = [m for m in texte.split() if m not in tous_les_mots_vides and len(m) > 2]
    return " ".join(mots)

# Chargement du modèle SentenceTransformer
nom_modele_st = 'paraphrase-multilingual-MiniLM-L12-v2'
print(f"Chargement du modèle {nom_modele_st} en cours...")
modele_nlp = SentenceTransformer(nom_modele_st)

# Récupération des fichiers PDF
fichiers_offres = glob.glob(os.path.join(dossier_offres, "*.pdf"))
fichiers_cv = glob.glob(os.path.join(dossier_cv, "*.pdf"))

if not fichiers_offres or not fichiers_cv:
    print("Erreur : Fichiers PDF manquants dans les dossiers de données")
    exit(1)

print(f"Traitement de {len(fichiers_offres)} offres et {len(fichiers_cv)} CV...")

# Extraction et nettoyage des contenus
donnees_offres = {os.path.basename(f): nettoyer_texte(extraire_texte_pdf(f)) for f in fichiers_offres}
donnees_cv = {os.path.basename(f): nettoyer_texte(extraire_texte_pdf(f)) for f in fichiers_cv}

# Filtrage par longueur de texte
donnees_offres = {k: v for k, v in donnees_offres.items() if len(v) > 50}
donnees_cv = {k: v for k, v in donnees_cv.items() if len(v) > 50}

# Liste des textes uniques pour calculer la PCA
textes_uniques = list(set(list(donnees_offres.values()) + list(donnees_cv.values())))
print(f"Génération des embeddings pour {len(textes_uniques)} documents...")
embeddings_uniques = modele_nlp.encode(textes_uniques)

# Réduction de dimension avec PCA
print("Calcul de la PCA 2D pour la visualisation...")
pca_modele = PCA(n_components=2)
pca_modele.fit(embeddings_uniques)

# Sauvegarde de l'objet PCA
chemin_pca = os.path.join(dossier_artifacts, 'pca.pickle')
with open(chemin_pca, 'wb') as f:
    pickle.dump(pca_modele, f)

# Construction du dataset final
liste_resultats = []
print("Génération du fichier CSV de référence...")

for nom_cv, texte_cv in donnees_cv.items():
    embedding_cv = modele_nlp.encode([texte_cv])
    coordonnees_cv = pca_modele.transform(embedding_cv)[0]
    
    for nom_offre, texte_offre in donnees_offres.items():
        embedding_offre = modele_nlp.encode([texte_offre])
        coordonnees_offre = pca_modele.transform(embedding_offre)[0]
        
        # Calcul du score de similarité
        score_similarite = float(cosine_similarity(embedding_cv, embedding_offre)[0][0])
        
        liste_resultats.append({
            'cv_text': texte_cv,
            'job_text': texte_offre,
            'similarity_score': score_similarite,
            'cv_pca_1': coordonnees_cv[0],
            'cv_pca_2': coordonnees_cv[1],
            'job_pca_1': coordonnees_offre[0],
            'job_pca_2': coordonnees_offre[1],
            'user_feedback': None,
            'timestamp': None
        })

df_reference = pd.DataFrame(liste_resultats)
df_reference.to_csv(chemin_csv_reference, index=False)

# Sauvegarde des informations du modèle
chemin_modele = os.path.join(dossier_artifacts, 'model.pickle')
with open(chemin_modele, 'wb') as f:
    pickle.dump({
        "model": modele_nlp,
        "name": nom_modele_st,
        "date_creation": pd.Timestamp.now().isoformat()
    }, f)

print(f"Progression terminée. Fichier généré : {chemin_csv_reference}")