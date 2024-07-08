from image_processing import FaceMeshModel, Image


def main():
    config_path = "config.json"
    model_path = "model.joblib"
    image_path = "image.jpg"
    
    face_mesh_model = FaceMeshModel(config_path)
    
    imagem = Image(image_path, face_mesh_model, model_path)
    
    imagem.predict()

if __name__ == "__main__":
    main()
