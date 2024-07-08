FROM python:3.12.3-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY image_processing.py scripts/image_processing.py
COPY main.py scripts/main.py

#COPY model.joblib app/model/model.joblib  # copiar o modelo para o container
# criar e copiar script para carregar e servir o modelo

CMD ["/bin/bash"]
