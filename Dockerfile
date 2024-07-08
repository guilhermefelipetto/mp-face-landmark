FROM python:3.12.3-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia os scripts Python para o diretório correto
COPY scripts/ /app/scripts/

# OPCIONAL: Descomente para copiar o modelo pré-treinado, se necessário
# COPY model.joblib /app/model/model.joblib

CMD ["python", "/app/scripts/main.py"]
