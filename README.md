# Projet MLOps CV Matching

Solution de matching CV / offres construite autour de FastAPI, Docker et d'un modele d'embedding MiniLM. Les scripts generent les artefacts (modele pickle, donnees de reference), l'API sert le scoring et collecte du feedback, et des dossiers sont prevus pour la webapp Streamlit et le monitoring Evidently.

## Architecture du depot
- scripts/ :
    - init_model.py : extrait le texte des PDFs dans data/resumes_corpus, calcule des embeddings MiniLM paraphrase-multilingual-MiniLM-L12-v2, genere data/ref_data.csv et artifacts/model.pickle.
    - init_prod_data.py : cree un CSV vide data/prod_data.csv pour stocker les feedbacks (cv_text, job_text, similarity_score, user_feedback, timestamp).
- serving/ : API FastAPI (api.py) avec endpoints /predict et /feedback, docker-compose pour exposer le service sur le port 8080, volumes data/ et artifacts/ montes dans le conteneur.
- data/ : corpus d'apprentissage (resumes_corpus/ avec des PDFs), ref_data.csv et prod_data.csv partages entre les services.
- artifacts/ : stockage du modele pickle charge par l'API.
- webapp/ : emplacement pour l'interface Streamlit (actuellement vide dans ce depot).
- reporting/ : emplacement pour Evidently/n8n (actuellement vide dans ce depot).

## Prerequis
- Docker et Docker Compose
- Python 3.10+ (pour executer les scripts en local)
- Au moins un fichier PDF dans data/resumes_corpus/

## Initialisation des artefacts (a faire avant Docker)
1) Installer les dependances locales si besoin :
     ```bash
     pip install sentence-transformers pandas scikit-learn pypdf
     ```
2) Generer les donnees de reference et le modele :
     ```bash
     python scripts/init_model.py
     ```
     Sorties : artifacts/model.pickle et data/ref_data.csv
3) Initialiser le fichier de production vide :
     ```bash
     python scripts/init_prod_data.py
     ```

## Lancer l'API de scoring
Depuis le dossier serving/ :
```bash
docker compose up --build
```
- Le service expose http://localhost:8080
- Les dossiers ../data et ../artifacts sont montes en volume pour partager ref_data.csv, prod_data.csv et model.pickle

## Endpoints principaux (serving/api.py)
- POST /predict :
    - Form-data : cv_file (PDF), job_text (string)
    - Reponse : similarity_score (float), match (bool, seuil 0.4), cv_text_extract (extrait du PDF)
- POST /feedback :
    - JSON : cv_text, job_text, similarity_score, user_feedback (bool)
    - Action : append dans /data/prod_data.csv puis log du trigger de reentrainement quand le nombre total de feedbacks est un multiple de 5 (RETRAIN_THRESHOLD_K)

## Notes et bonnes pratiques
- Ne pas commiter artifacts/model.pickle ni les donnees brutes (deja ignores dans .gitignore).
- Pour arreter et nettoyer : `docker compose down` depuis serving/.
- Les dossiers webapp/ et reporting/ sont des emplacements prevus pour la future UI Streamlit, l'automatisation n8n et le monitoring Evidently ; ils restent a completer.

---

# 🌿 Stratégie Git et Branchage

### Structure des branches

```
main
├── feature/scripts-pdf-preprocessing (Membre 1)
├── feature/scripts-incremental-retrain (Membre 1)
├── feature/api-health-endpoint (Membre 2)
├── feature/api-feedback-trigger (Membre 2)
├── feature/streamlit-webapp (Membre 3)
├── feature/n8n-workflow (Membre 4)
├── feature/docker-compose-orchestration (Membre 5)
└── feature/evidently-monitoring (Membre 5)
```

## Workflow Git standard

```bash
# 1. Créer une nouvelle branche depuis main
git checkout main
git pull origin main
git checkout -b feature/votre-feature

# 2. Développer et faire des commits réguliers
git add .
git commit -m "feat: description courte et explicite"

# 3. Pousser vers GitHub
git push origin feature/votre-feature

# 4. Synchro locale
git checkout main
git pull origin main
```

---

# 🎯 Prochaines Étapes & Répartition des Tâches (5 Étudiants)

## 👤 **Membre 1 : Data Scientist & Core ML**
**Responsable** : Amélioration du modèle et pipeline de données

### Tâches prioritaires :
- [ ] Tester et optimiser le preprocessing des PDFs dans `scripts/init_model.py`
  - Gestion des cas limites (PDFs corrompus, images)
  - Améliorer le nettoyage du texte (stopwords, normalisation)
- [ ] Évaluer d'autres modèles d'embedding (e.g., `sentence-transformers/all-MiniLM-L6-v2`)
- [ ] Implémenter une fonction de réentraînement incrémental dans `scripts/retrain_model.py`
  - Fusionner `data/ref_data.csv` + `data/prod_data.csv`
  - Régénérer les embeddings et sauvegarder dans `artifacts/model.pickle`
- [ ] Ajouter des métriques de performance (précision, recall) dans les logs

---

## 🔧 **Membre 2 : Backend Engineer (API FastAPI)**
**Responsable** : Évolution et stabilité de l'API

### Tâches prioritaires :
- [ ] Implémenter le trigger de réentraînement dans `serving/api.py`
  - Compter les feedbacks dans `data/prod_data.csv`
  - Déclencher `python scripts/retrain_model.py` quand nombre de feedbacks ≡ 0 [mod 5]
- [ ] Ajouter une route `GET /health` pour monitoring
- [ ] Ajouter une route `GET /model-info` retournant la date d'entraînement et nombre d'embeddings
- [ ] Implémenter la gestion d'erreurs robuste (timeout pour les grands PDFs, logging)
- [ ] Écrire des tests unitaires pour `/predict` et `/feedback`
- [ ] Documenter l'API (docstrings OpenAPI)

---

## 🎨 **Membre 3 : Frontend Engineer (Streamlit)**
**Responsable** : Interface utilisateur interactive

### Tâches prioritaires :
- [ ] Créer `webapp/app.py` (interface Streamlit)
  - Formulaire d'upload de CV (PDF)
  - Input textarea pour la description du poste
  - Bouton "Calculer la similarité"
- [ ] Connecter à l'API de scoring
  - Appel POST à `http://serving-api:8080/predict`
  - Afficher le score de similarité et le statut (match/no-match)
  - Afficher l'extrait de texte du CV
- [ ] Ajouter un système de feedback
  - Boutons "Pertinent" / "Non pertinent"
  - Appel POST à `http://serving-api:8080/feedback`
  - Message de confirmation
- [ ] Créer `webapp/requirements.txt` et `webapp/Dockerfile`
- [ ] Tester l'intégration Docker (port 8081)

---

## 🤖 **Membre 4 : AI Automation Specialist (n8n/LLM)**
**Responsable** : Workflow d'automatisation et explications IA

### Tâches prioritaires :
- [ ] Configurer n8n dans `docker-compose.yml` (port 5678)
- [ ] Créer un workflow n8n qui :
  - Reçoit un Webhook depuis Streamlit avec (cv_text, job_text, similarity_score)
  - Appelle une LLM (Gemini API ou OpenAI) pour générer une explication
  - Exemple de prompt : *"Le CV matche à 75% avec l'offre. Explique en 2-3 phrases pourquoi."*
  - Envoie un email au candidat/recruteur avec le résultat
- [ ] Configurer l'authentification n8n (clés API Gemini/OpenAI, SMTP)
- [ ] Créer une documentation n8n du workflow (captures d'écran)
- [ ] Tester l'intégration bout-en-bout (Streamlit → n8n → Email)

---

## 🏗 **Membre 5 : DevOps & Monitoring Lead**
**Responsable** : Infrastructure, monitoring et qualité des données

### Tâches prioritaires :
- [ ] Mettre à jour `docker-compose.yml` principal (à la racine)
  - Services : serving, webapp, reporting, n8n
  - Networks partagés et volumes montés
  - Variables d'environnement (API keys, thresholds)
- [ ] Créer `reporting/project.py` (Evidently)
  - Charger `data/prod_data.csv` et `data/ref_data.csv`
  - Détecter le Data Drift (distribution des scores de similarité)
  - Générer des rapports HTML dans `reporting/reports/`
- [ ] Implémenter un script `scripts/monitor.py`
  - Surveiller la taille de `prod_data.csv`
  - Logger les anomalies (PDFs mal traités, scores aberrants)
- [ ] Configurer `reporting/requirements.txt` et `reporting/Dockerfile`
- [ ] Rédiger la documentation DevOps (logs, debugging, scaling)
- [ ] Mettre en place un `.env.example` pour les variables sensibles

---

## 🔄 Communication Inter-Équipes

| Membre | Dépend de | Fournit à |
|--------|-----------|-----------|
| **1 (Data)** | - | 2 (trigger réentraînement), 5 (métriques) |
| **2 (Backend)** | 1 | 3 (endpoint /predict), 4 (logs) |
| **3 (Frontend)** | 2 | 4 (Webhook avec résultats) |
| **4 (Automation)** | 2, 3 | 5 (logs emails) |
| **5 (DevOps)** | Tous | Tous (infrastructure + monitoring) |

---

## 🚀 Timeline Recommandée

**Semaine 1** : Membres 1 & 2 finalisent le modèle et l'API, Membre 5 prépare l'infra.

**Semaine 2** : Membre 3 développe la webapp, Membre 4 configure n8n.

**Semaine 3** : Intégration complète, tests et monitoring, Membre 5 finalise la documentation.

---

## ✅ Checklist de Validation Globale

- [ ] Tous les scripts locaux exécutés sans erreur (`init_model.py`, `init_prod_data.py`)
- [ ] API démarrée et endpoints testés (curl ou Postman)
- [ ] Webapp Streamlit fonctionnelle et connectée à l'API
- [ ] Workflow n8n exécuté avec succès (Webhook → LLM → Email)
- [ ] Monitoring Evidently génère un rapport
- [ ] `docker compose up --build` fonctionne en une seule commande
- [ ] README à jour avec les instructions finales