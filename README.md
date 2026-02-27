# Projet de Matching de CV et Offres d'Emploi

Ce projet a pour objectif d'automatiser le processus de matching entre des CV (format PDF) et des descriptions de postes en utilisant des techniques de Traitement du Langage Naturel (NLP) et une boucle de feedback avec [n8n](https://github.com/n8n-io/n8n).

---

## 👥 État du Projet et Responsabilités

### Membre 1 (Albin) : Data Scientist & Lead
- **Statut** : ✅ Terminé
- **Réalisations** :
  - [x] Extraction de texte via `pdfplumber`.
  - [x] Nettoyage sémantique (regex, stop-words français/anglais).
  - [x] Initialisation du modèle `SentenceTransformer`.
  - [x] Calcul des embeddings et projection PCA 2D.
  - [x] Export des données de référence vers `data/ref_data.csv`.
- **Commande** : `python scripts/init_model.py`

### Membre 2 : Backend Engineer
- **Statut** : 🛠️ En cours (Squelette de test)
- **Objectifs à venir** :
  - [ ] Intégration de la logique de réentraînement dans `scripts/retrain_model.py`.
  - [ ] Développement des endpoints d'API complets (`/jobs`, `/predict` réel).
  - [ ] Gestion de la base de données de production `data/prod_data.csv`.
- **Commande actuelle** : `python serving/api.py`

### Membre 3 : Frontend Engineer
- **Statut** : 🛠️ En cours (Squelette de test)
- **Objectifs à venir** :
  - [ ] Dashboard complet de visualisation des scores.
  - [ ] Graphique interactif PCA 2D via Plotly.
  - [ ] Système de gestion et d'historique des analyses.
- **Commande actuelle** : `streamlit run webapp/app.py`

### Membre 4 : Spécialiste AI Automation
- **Statut** : 🛠️ En cours (Squelette de test)
- **Objectifs à venir** :
  - [ ] Configuration du workflow n8n.
  - [ ] Test de la boucle de feedback Gemini -> Gmail.
- **Fonctionnement** : n8n reçoit les données, Gemini explique le score, validation par email.

### Membre 5 (Mustapha) : DevOps & Monitoring
- **Statut** : 🛠️ En cours (Test)
- **Objectifs à venir** :
  - [ ] Analyse de drift avec `evidently` dans `reporting/project.py`.
  - [ ] Préparation des Dockerfiles (Backend & Frontend).
  - [ ] Optimisation de l'orchestration `docker-compose.yml`.
  - [ ] Déploiement CI/CD.

---

## 🚀 Guide de Démarrage

### 1. Préparation de l'environnement

**Sur Windows (PowerShell) :**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Sur Linux / macOS :**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Lancement des Services (Test n8n)
Pour tester la boucle de matching et feedback avec n8n, lancez les deux commandes suivantes dans des terminaux séparés :

**A. Serveur Backend (FastAPI)**
```bash
docker compose -f serving/docker-compose.yml up
```

**B. Application Frontend (Streamlit)**
```bash
docker compose -f webapp/docker-compose.yml up
```

**C. Monitoring (Evidently)**
```bash
docker compose -f monitoring/docker-compose.yml up
```

---

### 3. Déploiement avec Docker
Si vous préférez utiliser Docker (recommandé pour la production) :

**Build et lancement** :
```bash
docker compose up --build
```
L'API sera accessible sur le port **8000** et Streamlit sur le port **8501**.

---

## 📂 Structure du Dossier

```text
projet_data/
├── artifacts/          # Objets pickle du modèle et de la PCA + rapports
├── data/               # Corpus de documents (CV/Jobs) et fichiers CSV
├── scripts/            # Logique ML (Init & Placeholder de réentraînement)
├── serving/            # Backend FastAPI (Simulateur n8n)
├── webapp/             # Dashboard Streamlit (Interface de test)
├── reporting/          # Placeholder Monitoring (DevOps)
├── Dockerfile.backend  # Configuration Docker Backend
├── Dockerfile.frontend # Configuration Docker Frontend
└── docker-compose.yml  # Orchestration multi-conteneurs
```



docker volume create n8n_data
docker run -it --rm --name n8n -p 5678:5678 -v n8n_data:/home/node/.n8n docker.n8n.io/n8nio/n8n


commande de génération de rapport:
```
docker exec evidently-reporting python project.py
```

commande pour lancer n8n  correctement:
```
docker run -d --name n8n -p 5678:5678 --network mon_reseau_projet -v n8n_data:/home/node/.n8n docker.n8n.io/n8nio/n8n
```