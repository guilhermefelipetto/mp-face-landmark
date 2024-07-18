import os
import cv2
import json
import numpy as np
import mediapipe as mp
from joblib import load

class FaceMeshModel:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = json.load(file)
        
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=self.config["static_image_mode"],
            max_num_faces=self.config["max_num_faces"],
            min_detection_confidence=self.config["min_detection_confidence"]
        )
        self.points = []
        self.image = None
        self.reference_distance = None

    def calculate_distance(self, idx1, idx2):
        if not self.points:
            raise ValueError("No landmarks available. Process an image first.")
        p1 = self.points[idx1]
        p2 = self.points[idx2]
        return np.linalg.norm(np.array(p1) - np.array(p2))
    
    def save_landmark_distances(self, output_file):
        if self.reference_distance is None:
            self.reference_distance = self.calculate_distance(10, 152)
        
        directory = "/workdir/scripts/data/default_landmark_distances"
        os.makedirs(directory, exist_ok=True)
        output_path = os.path.join(directory, output_file)
        
        with open(output_path, 'w') as file:
            for i in range(len(self.points) - 1):
                dist = self.calculate_distance(i, i + 1) / self.reference_distance
                file.write(f'{i} {i+1} {dist}\n')

    def calculate_and_draw_distance(self, idx1, idx2, save=False, measure_type='None'):
        if not self.points:
            raise ValueError("No landmarks available. Process an image first.")
        
        if self.reference_distance is None:
            self.reference_distance = self.calculate_distance(10, 152)
        
        dist = self.calculate_distance(idx1, idx2) / self.reference_distance
        cv2.line(self.image, self.points[idx1], self.points[idx2], (0, 255, 0), 2)
        cv2.putText(self.image, f"{dist:.2f}", ((self.points[idx1][0] + self.points[idx2][0]) // 2, (self.points[idx1][1] + self.points[idx2][1]) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if save:
            directory = "/workdir/scripts/data/specific_distances"
            os.makedirs(directory, exist_ok=True)

            last_folder = os.path.basename(os.path.dirname(self.image_path))
            last_folder = last_folder.replace(" ", "_").lower()

            image_basename = os.path.splitext(os.path.basename(self.image_path))[0]
            output_filename = f"{directory}/{last_folder}_{image_basename}_distances.txt"

            measure_type = measure_type.replace(' ', '_').strip()
            with open(output_filename, 'a') as file:
                file.write(f'{idx1} {idx2} {dist} {measure_type}\n')

        return dist

    def process_image(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.points = [(int(lm.x * self.image.shape[1]), int(lm.y * self.image.shape[0])) for lm in face_landmarks.landmark]
                
                if self.config["draw_landmarks"]:
                    mp.solutions.drawing_utils.draw_landmarks(
                        image=self.image,
                        landmark_list=face_landmarks,
                        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION if self.config["draw_connections"] else None,
                        landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(**self.config["landmark_drawing_spec"]),
                        connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(**self.config["connection_drawing_spec"])
                    )
                
                if self.config["draw_numbers"]:
                    for idx, point in enumerate(self.points):
                        cv2.putText(self.image, str(idx), (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

        self.reference_distance = self.calculate_distance(10, 152)
        image_basename = os.path.basename(image_path).split('.')[0]
        output_file = f"{image_basename}_landmarks_distances.txt"
        self.save_landmark_distances(output_file)

    def display_or_save_image(self, save_path=None):
        if save_path:
            cv2.imwrite(save_path, self.image)
        else:
            cv2.imshow('Facial Landmarks', self.image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


class Image:
    def __init__(self, path, facemesh_model, model_path):
        self.path = path
        self.modelo_face_mesh = facemesh_model
        self.modelo_predicao = load(model_path)
        self.imagem = None
        self.features = None

    def carregar_imagem(self):
        self.imagem = cv2.imread(self.path)
        if self.imagem is not None:
            print("Imagem carregada com sucesso!")
            return True
        else:
            print("Erro ao carregar imagem. Verifique o caminho.")
            return False

    def predict(self):
        if self.carregar_imagem():
            self.modelo_face_mesh.process_image(self.path)

            self.features = [
                self.modelo_face_mesh.calculate_and_draw_distance(10, 152),
                self.modelo_face_mesh.calculate_and_draw_distance(1, 152),
                self.modelo_face_mesh.calculate_and_draw_distance(1, 10),
                self.modelo_face_mesh.calculate_and_draw_distance(1, 361),
                self.modelo_face_mesh.calculate_and_draw_distance(1, 132),
                self.modelo_face_mesh.calculate_and_draw_distance(1, 365),
                self.modelo_face_mesh.calculate_and_draw_distance(1, 150),
                self.modelo_face_mesh.calculate_and_draw_distance(334, 443),
                self.modelo_face_mesh.calculate_and_draw_distance(282, 334),
                self.modelo_face_mesh.calculate_and_draw_distance(105, 223),
                self.modelo_face_mesh.calculate_and_draw_distance(52, 105),
                self.modelo_face_mesh.calculate_and_draw_distance(54, 284),
                self.modelo_face_mesh.calculate_and_draw_distance(9, 10),
                self.modelo_face_mesh.calculate_and_draw_distance(133, 463),
                self.modelo_face_mesh.calculate_and_draw_distance(1, 159),
                self.modelo_face_mesh.calculate_and_draw_distance(1, 386),
                self.modelo_face_mesh.calculate_and_draw_distance(159, 386),
                self.modelo_face_mesh.calculate_and_draw_distance(133, 33),
                self.modelo_face_mesh.calculate_and_draw_distance(263, 362),
                self.modelo_face_mesh.calculate_and_draw_distance(1, 278),
                self.modelo_face_mesh.calculate_and_draw_distance(1, 48),
                self.modelo_face_mesh.calculate_and_draw_distance(0, 1),
                self.modelo_face_mesh.calculate_and_draw_distance(1, 16),
                self.modelo_face_mesh.calculate_and_draw_distance(1, 61),
                self.modelo_face_mesh.calculate_and_draw_distance(1, 291),
                self.modelo_face_mesh.calculate_and_draw_distance(0, 16),
                self.modelo_face_mesh.calculate_and_draw_distance(10, 33),
                self.modelo_face_mesh.calculate_and_draw_distance(10, 133),
                self.modelo_face_mesh.calculate_and_draw_distance(10, 263),
                self.modelo_face_mesh.calculate_and_draw_distance(10, 362),
                self.modelo_face_mesh.calculate_and_draw_distance(33, 152),
                self.modelo_face_mesh.calculate_and_draw_distance(133, 152),
                self.modelo_face_mesh.calculate_and_draw_distance(152, 263),
                self.modelo_face_mesh.calculate_and_draw_distance(152, 362)
            ]

            # Obtenha as probabilidades das previsões
            probabilidades = self.modelo_predicao.predict_proba([self.features])[0]
            classe_predita = self.modelo_predicao.predict([self.features])[0]
            
            # Encontre o índice da classe com a maior probabilidade
            max_index = np.argmax(probabilidades)
            max_probability = probabilidades[max_index] * 100

            print(f"Predição: Elemento: {classe_predita}, {max_probability:.2f}%")
            return classe_predita, max_probability
        else:
            print("Imagem não está carregada ou erro na carga.")
            return None
