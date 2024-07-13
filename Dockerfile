FROM python:3.12.3-slim

WORKDIR /workdir

RUN apt-get update && apt-get install -y libpq-dev
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get update && apt-get install -y libglib2.0-0

COPY requirements.txt .
RUN pip --default-timeout=100 install --no-cache-dir -r requirements.txt

RUN mkdir -p ./scripts/data/specific_distances
RUN mkdir -p ./scripts/data/default_landmark_distances
RUN mkdir -p ./model/

COPY model/model.joblib ./model/model.joblib
COPY scripts/main.py ./scripts/main.py
COPY scripts/config.json ./scripts/config.json

COPY scripts/image_processing.py ./scripts/image_processing.py
COPY scripts/process_data.py ./scripts/process_data.py

COPY scripts/load_db.py ./scripts/load_db.py
COPY scripts/process_new_image.py ./scripts/process_new_image.py

COPY scripts/train.py ./scripts/train.py
COPY scripts/predict.py ./scripts/predict.py

COPY scripts/database/db.py ./scripts/database/db.py
COPY webapp/ ./webapp/