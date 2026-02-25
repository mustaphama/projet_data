# On utilise une image Python légère
FROM python:3.10-slim

# On définit le dossier de travail dans le conteneur
WORKDIR /app

# On copie le fichier des dépendances et on les installe
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# On copie tout le reste du code de la webapp (app.py, etc.)
COPY . .

# On expose le port par défaut de Streamlit
EXPOSE 8501

# Commande pour démarrer l'interface Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]