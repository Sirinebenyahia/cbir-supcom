# Image de base légère avec Python 3.11
FROM python:3.11-slim

# Bibliothèques système nécessaires à OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dépendances Python (PyTorch en version CPU)
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Code de l'application (les données sont montées en volumes, voir docker-compose.yml)
COPY backend/ backend/
COPY frontend/ frontend/
COPY eval/ eval/
COPY .streamlit/ .streamlit/

EXPOSE 8000 8501

# Démarre le serveur FastAPI puis le client Streamlit
CMD ["sh", "-c", "uvicorn api:app --app-dir backend --host 0.0.0.0 --port 8000 & sleep 25 && streamlit run frontend/app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true"]