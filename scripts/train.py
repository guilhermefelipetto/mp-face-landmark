import os
import re
import pandas as pd
import seaborn as sns
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sqlalchemy import create_engine
from joblib import dump

load_dotenv()

def get_next_filename(directory, base_name, extension):
    pattern = re.compile(rf'{re.escape(base_name)}_(\d+){re.escape(extension)}')
    existing_files = [f for f in os.listdir(directory) if pattern.match(f)]
    
    if not existing_files:
        return f'{base_name}_01{extension}'
    
    numbers = [int(pattern.match(f).group(1)) for f in existing_files]
    next_number = max(numbers) + 1

    return f'{base_name}_{next_number:02}{extension}'

def train_model(MODEL_NAME, DB_URL, TABLE_NAME, N_ESTIMATORS=200, MAX_FEATURES='sqrt', TEST_SIZE=0.3, MAX_DEPTH=10, RANDOM_STATE=42, MIN_SAMPLES_SPLIT=2, BOOTSTRAP=True, MIN_SAMPLES_LEAF=1, SHOW_INFO=False):
    engine = create_engine(DB_URL)
    df = pd.read_sql_table(TABLE_NAME, con=engine)

    X = df.drop('type', axis=1)
    y = df['type']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI')
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    mlflow.set_experiment('MODEL_MTCRandomForest')
    with mlflow.start_run():
        
        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, 
            max_depth=MAX_DEPTH,
            min_samples_split=MIN_SAMPLES_SPLIT,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            max_features=MAX_FEATURES,
            bootstrap=BOOTSTRAP,
            random_state=RANDOM_STATE
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("max_depth", MAX_DEPTH)
        mlflow.log_param("min_samples_split", MIN_SAMPLES_SPLIT)
        mlflow.log_param("min_samples_leaf", MIN_SAMPLES_LEAF)
        mlflow.log_param("max_features", MAX_FEATURES)
        mlflow.log_param("bootstrap", BOOTSTRAP)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.set_tag("framework", "Scikit-learn")
        mlflow.set_tag("data_split", "train_test_split")
        mlflow.set_tag("features_shape", str(X_train.shape))

        report = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_metric("weighted_precision", report['weighted avg']['precision'])
        mlflow.log_metric("weighted_recall", report['weighted avg']['recall'])
        mlflow.log_metric("weighted_f1_score", report['weighted avg']['f1-score'])

        mlflow.sklearn.log_model(model, "random_forest_classifier")
        dump(model, f'/workdir/model/{MODEL_NAME}_model.joblib')

        conf_matrix = confusion_matrix(y_test, y_pred)
        class_labels = df['type'].unique()
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')

        artifacts_dir = '/workdir/artifacts'
        if not os.path.exists(artifacts_dir):
            os.makedirs(artifacts_dir)
        
        base_name = 'confusion_matrix'
        extension = '.jpg'
        filename = get_next_filename(artifacts_dir, base_name, extension)
        file_path = os.path.join(artifacts_dir, filename)

        plt.savefig(file_path)
        plt.close()

        mlflow.log_artifact(file_path)

        if SHOW_INFO:
            print(f'Acurácia do modelo: {accuracy * 100:.2f}%')
            print(classification_report(y_test, y_pred))
