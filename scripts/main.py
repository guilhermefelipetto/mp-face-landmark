import os
import load_db
import run_train
import predict
from flask import Flask, render_template, request, jsonify, url_for, redirect
from process_new_image import process_new_image
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from database import db

app = Flask(__name__, static_folder='/workdir/webapp/static', template_folder='/workdir/webapp/templates')
load_dotenv()

app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

app.config['MLFLOW_DATABASE_URI'] = os.environ.get('MLFLOW_DATABASE_URI')
app.config['TABLE_NAME'] = os.environ.get('TABLE_NAME')
app.config['N_ESTIMATORS'] = os.environ.get('N_ESTIMATORS')
app.config['MAX_FEATURES'] = os.environ.get('MAX_FEATURES')
app.config['TEST_SIZE'] = os.environ.get('TEST_SIZE')
app.config['MAX_DEPTH'] = os.environ.get('MAX_DEPTH')
app.config['RANDOM_STATE'] = os.environ.get('RANDOM_STATE')
app.config['MIN_SAMPLES_SPLIT'] = os.environ.get('MIN_SAMPLES_SPLIT')
app.config['BOOTSTRAP'] = os.environ.get('BOOTSTRAP')
app.config['MIN_SAMPLES_LEAF'] = os.environ.get('MIN_SAMPLES_LEAF')
app.config['SHOW_INFO'] = os.environ.get('SHOW_INFO')

app.config['UPLOAD_FOLDER'] = '/workdir/webapp/static/images'
app.config['PREDICT_FOLDER'] = '/workdir/scripts/predict_image'

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload_image', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        image_type = request.form.get('type')

        if not image_type or image_type not in ['agua', 'fogo', 'terra', 'metal', 'madeira']:
            return jsonify({'error': 'Tipo inválido ou não selecionado'}), 400
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], image_type, filename)
            file.save(file_path)

            return redirect(url_for('process_image', image_type=image_type, filename=filename))
        
        else:
            return jsonify({'error': 'File type not allowed'}), 400


@app.route('/process_image/<image_type>/<filename>', methods=['GET', 'POST'])
def process_image(image_type, filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_type, filename)
    
    try:
        processed_data = process_new_image(image_path=image_path)
        
        _, _ = db.create_database_if_not_exists(url=app.config['SQLALCHEMY_DATABASE_URI'])
        
        train_data_params = {
            'root_dir': '/workdir/webapp/static/images',
            'config_path': '/workdir/scripts/config.json'
        }

        load_db.func_load_db(train_data_params=train_data_params, db_url=app.config['SQLALCHEMY_DATABASE_URI'])

        for root, _, files in os.walk('/workdir/webapp/static/images'):
            if files:
                for file in files:
                    if file.endswith('.jpg'):
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        else:
                            print(f"Arquivo não encontrado: {file_path}")
            else:
                print(f"Sem arquivos na pasta: {root}")

        for root, _, files in os.walk('/workdir/scripts/data/specific_distances'):
            if files:
                for file in files:
                    if file.endswith('.txt'):
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        else:
                            print(f"Arquivo não encontrado: {file_path}")
            else:
                print(f"Sem arquivos na pasta: {root}")

        for root, _, files in os.walk('/workdir/scripts/data'):
            if files:
                for file in files:
                    if file.endswith('.csv'):
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        else:
                            print(f"Arquivo não encontrado: {file_path}")
            else:
                print(f"Sem arquivos na pasta: {root}")
        
        return render_template('success.html', image_type=image_type, filename=filename, processed_data=processed_data)
    
    except Exception as e:
        app.logger.error(f'Erro ao processar imagem: {str(e)}')
        
        return render_template('error.html', error_message=str(e))


@app.route('/train_model', methods=['GET', 'POST'])
def train_model():  # POR ALGUM MOTIVO A CONFUSION MATRIX NAO ESTA SENDO SALVA NO MLFLOW COMO ARTEFATO
    try:
        run_train.run(
            MODEL_NAME='latest',
            DB_URL=app.config['MLFLOW_DATABASE_URI'],
            TABLE_NAME=app.config['TABLE_NAME'],
            N_ESTIMATORS=app.config['N_ESTIMATORS'],
            MAX_FEATURES=app.config['MAX_FEATURES'],
            TEST_SIZE=app.config['TEST_SIZE'],
            MAX_DEPTH=app.config['MAX_DEPTH'],
            RANDOM_STATE=app.config['RANDOM_STATE'],
            MIN_SAMPLES_SPLIT=app.config['MIN_SAMPLES_SPLIT'],
            BOOTSTRAP=app.config['BOOTSTRAP'].lower() == 'true',
            MIN_SAMPLES_LEAF=app.config['MIN_SAMPLES_LEAF'],
            SHOW_INFO=app.config['SHOW_INFO'].lower() == 'true'
        )
    
    except Exception as e:
        print('ERRO:', e)
    
    return render_template('train.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['PREDICT_FOLDER'], filename)
        file.save(file_path)

        try:
            params = {
                'model_path': '/workdir/model/model.joblib',
                'config_path': '/workdir/scripts/config.json',
                'image_path': file_path
            }
            classe_predita, max_probability = predict.predict(params)

            return render_template('predict.html', classe_predita=classe_predita, max_probability=max_probability * 100)
        
        except Exception as e:
            app.logger.error(f'Erro ao processar imagem: {str(e)}')

            return render_template('error.html', error_message=str(e))
    else:
        return jsonify({'error': 'File type not allowed or no file provided'}), 400


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5002)
