import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from joblib import dump

N_ESTIMATORS = 200
MAX_FEATURES = 'sqrt'
TEST_SIZE = 0.3
MAX_DEPTH = 10
RANDOM_STATE = 42
MIN_SAMPLES_SPLIT = 2
BOOTSTRAP = True
MIN_SAMPLES_LEAF = 1

df = pd.read_csv('data.csv')

X = df.drop('Type', axis=1)
y = df['Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

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
    dump(model, 'model.joblib')

    conf_matrix = confusion_matrix(y_test, y_pred)
    class_labels = df['Type'].unique()

    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.jpg')
    plt.close()

    mlflow.log_artifact('confusion_matrix.jpg')

    print(f'Acurácia do modelo: {accuracy * 100:.2f}%')
    print(classification_report(y_test, y_pred))
