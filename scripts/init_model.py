import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_classification

# 1. on genère un dataset pour l'exemple
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
df['target'] = y

# 2. Préparation des objets (Scaler + PCA + Modèle)
scaler = StandardScaler()
pca = PCA(n_components=3) # 3 dimensions principales
model = LogisticRegression()

# 3. Entraînement
X_scaled = scaler.fit_transform(X)
X_pca = pca.fit_transform(X_scaled)
model.fit(X_pca, y)

# 4. Sauvegarde du fichier ref_data.csv
ref_data = pd.DataFrame(X_pca, columns=['PCA 1', 'PCA 2', 'PCA 3'])
ref_data['target'] = y
ref_data['target'] = ref_data['target'].astype(bool)
ref_data.to_csv('../data/ref_data.csv', index=False)
print("✅ data/ref_data.csv généré")

# 5. Sauvegarde des artefacts
with open('../artifacts/scaler.pickle', 'wb') as f:
    pickle.dump(scaler, f)

with open('../artifacts/pca.pickle', 'wb') as f:
    pickle.dump(pca, f)
    
with open('../artifacts/model.pickle', 'wb') as f:
    pickle.dump(model, f)

print("✅ Artifacts (model, scaler, pca) sauvegardés dans artifacts/")