# On passe à Python 3.11 pour forcer Docker à créer un environnement 100% vierge
FROM python:3.11-slim

WORKDIR /app

RUN pip install --upgrade pip

# On fixe la version très récente 0.4.33 qui gère parfaitement les conflits Pydantic
RUN pip install --no-cache-dir evidently==0.4.33 pandas scikit-learn pydantic>=2.0

COPY project.py .

RUN mkdir workspace

CMD python project.py && evidently ui --workspace workspace --host 0.0.0.0 --port 8082