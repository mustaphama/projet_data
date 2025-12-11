import pandas as pd
import os

# Chemin vers data/prod_data.csv
path = os.path.join(os.path.dirname(__file__), '../data/prod_data.csv')

# Colonnes attendues (Doit matcher ref_data.csv + la colonne 'user_feedback')
# On simplifie ici pour stocker l'essentiel
columns = ['cv_text', 'job_text', 'similarity_score', 'user_feedback', 'timestamp']
df = pd.DataFrame(columns=columns)
df.to_csv(path, index=False)
print(f"✅ Fichier de production initialisé : {path}")