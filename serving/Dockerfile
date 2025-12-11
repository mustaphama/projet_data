# serving/Dockerfile

# Indique au conteneur de baser son environnement sur l'image Docker de Python 3.10
FROM python:3.10-slim

# Défini le directory de travail
WORKDIR /app

# Copie le code de votre API et le fichier de requirements
COPY api.py .
COPY requirements.txt .

# Met à jour pip et installe les packages indiquées
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Héberge l'API sur le port 8080 (IP 0.0.0.0 pour être accessible hors conteneur)
CMD uvicorn api:app --host 0.0.0.0 --port 8080