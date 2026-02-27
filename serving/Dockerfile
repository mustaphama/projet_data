FROM python:3.10-slim

WORKDIR /app

COPY api.py .
COPY requirements.txt .

RUN pip3 install --upgrade pip
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install --default-timeout=1000 -r requirements.txt

CMD uvicorn api:app --host 0.0.0.0 --port 8080