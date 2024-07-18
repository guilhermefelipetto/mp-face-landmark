from train import train_model
import os

# train_params = {
#     'MODEL_NAME': 'teste',
#     'DB_URL': os.environ.get('MLFLOW_DATABASE_URI'),
#     'TABLE_NAME': os.environ.get('TABLE_NAME'),
#     'N_ESTIMATORS': int(os.environ.get('N_ESTIMATORS')),
#     'MAX_FEATURES': os.environ.get('MAX_FEATURES'),
#     'TEST_SIZE': float(os.environ.get('TEST_SIZE')),
#     'MAX_DEPTH': int(os.environ.get('MAX_DEPTH')),
#     'RANDOM_STATE': int(os.environ.get('RANDOM_STATE')),
#     'MIN_SAMPLES_SPLIT': int(os.environ.get('MIN_SAMPLES_SPLIT')),
#     'BOOTSTRAP': os.environ.get('BOOTSTRAP') == 'True',
#     'MIN_SAMPLES_LEAF': int(os.environ.get('MIN_SAMPLES_LEAF')),
#     'SHOW_INFO': os.environ.get('SHOW_INFO') == 'True'
# }

def run(MODEL_NAME, DB_URL, TABLE_NAME, N_ESTIMATORS, MAX_FEATURES, TEST_SIZE, MAX_DEPTH, RANDOM_STATE, MIN_SAMPLES_SPLIT, BOOTSTRAP, MIN_SAMPLES_LEAF, SHOW_INFO):
    train_params = {
        'MODEL_NAME': MODEL_NAME,
        'DB_URL': DB_URL,
        'TABLE_NAME': TABLE_NAME,
        'N_ESTIMATORS': int(N_ESTIMATORS),
        'MAX_FEATURES': MAX_FEATURES,
        'TEST_SIZE': float(TEST_SIZE),
        'MAX_DEPTH': int(MAX_DEPTH),
        'RANDOM_STATE': int(RANDOM_STATE),
        'MIN_SAMPLES_SPLIT': int(MIN_SAMPLES_SPLIT),
        'BOOTSTRAP': BOOTSTRAP,
        'MIN_SAMPLES_LEAF': int(MIN_SAMPLES_LEAF),
        'SHOW_INFO': SHOW_INFO
    }
    
    train_model(**train_params)
