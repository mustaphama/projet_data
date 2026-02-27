FROM python:3.11-slim

WORKDIR /app

RUN pip install --upgrade pip

RUN pip install --no-cache-dir evidently==0.4.33 pandas scikit-learn pydantic>=2.0

COPY project.py .

RUN mkdir workspace

CMD python project.py && evidently ui --workspace workspace --host 0.0.0.0 --port 8082