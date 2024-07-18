from image_processing import FaceMeshModel, Image


def predict(image_path, model_path='/workdir/model/model.joblib', config_path='/workdir/scripts/config.json'):    
    face_mesh_model = FaceMeshModel(config_path)
    
    imagem = Image(image_path, face_mesh_model, model_path)
    
    classe_predita, max_probability = imagem.predict()
    
    return classe_predita, max_probability


# predict(image_path = "images_test/aaa.jpg", model_path = "model/model.joblib", config_path = "scripts/config.json",)
