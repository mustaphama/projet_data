# 🧠 End-to-End AI Matching Platform (MLOps Engine)

Ce projet est le **moteur d'Intelligence Artificielle et l'infrastructure MLOps** permettant d'automatiser le processus de matching sémantique entre des CV (format PDF) et des offres d'emploi. Il utilise des techniques avancées de NLP (SentenceTransformers), une API de serving robuste, du monitoring de dérive (data drift) et une boucle d'automatisation des retours.

> 💡 **Interface Web / Application Client**  
> Ce dépôt se concentre sur le backend ML et l'orchestration. L'application principale (Django/React) qui consomme cette API est disponible ici :[**Albin0903/codev**](https://github.com/Albin0903/codev).

---

## ✨ Architecture & Fonctionnalités

Ce pipeline couvre l'intégralité du cycle de vie du modèle ML en production :

*   **Pipeline NLP & Embeddings** : Extraction de texte (`pdfplumber`), nettoyage sémantique (regex, stop-words), et vectorisation via `SentenceTransformers` avec projection PCA 2D.
*   **Model Serving** : API REST performante développée avec **FastAPI** pour exposer les prédictions et gérer la base de données en production.
*   **Monitoring MLOps** : Suivi continu de la performance et détection de la dérive des données (Data Drift) via **Evidently**.
*   **Boucle de Feedback Automatisée** : Intégration de workflows **n8n** couplés à l'API LLM Gemini pour générer des explications sur les scores de matching et valider les actions par email.
*   **Infrastructure & Conteneurisation** : Architecture microservices entièrement dockerisée (Backend, Streamlit UI, Reporting) orchestrée via `docker-compose`.

---

## 🚀 Guide de Démarrage

### 1. Déploiement via Docker (Recommandé)

L'ensemble de la stack technique est packagé pour un déploiement rapide et isolé.

**Build et lancement de l'API et de l'interface de test :**
```bash
docker compose up --build -d