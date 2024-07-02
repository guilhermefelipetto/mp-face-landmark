import cv2
import mediapipe as mp
import numpy as np
import json
import os

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

    def calculate_distance(self, idx1, idx2):
        if not self.points:
            raise ValueError("No landmarks available. Process an image first.")
        p1 = self.points[idx1]
        p2 = self.points[idx2]
        return np.linalg.norm(np.array(p1) - np.array(p2))
    
    def save_landmark_distances(self, output_file):
        directory = "default_landmark_distances"
        os.makedirs(directory, exist_ok=True)
        output_path = os.path.join(directory, output_file)
        
        with open(output_path, 'w') as file:
            for i in range(len(self.points) - 1):
                dist = self.calculate_distance(i, i + 1)
                file.write(f'{i} {i+1} {dist}\n')

    def calculate_and_draw_distance(self, idx1, idx2, save=False, measure_type='None'):
        if not self.points:
            raise ValueError("No landmarks available. Process an image first.")
        dist = self.calculate_distance(idx1, idx2)
        cv2.line(self.image, self.points[idx1], self.points[idx2], (0, 255, 0), 2)
        cv2.putText(self.image, f"{dist:.2f}", ((self.points[idx1][0] + self.points[idx2][0]) // 2, (self.points[idx1][1] + self.points[idx2][1]) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if save:
            directory = "specific_distances"
            os.makedirs(directory, exist_ok=True)
            image_basename = os.path.splitext(os.path.basename(self.image_path))[0]
            output_filename = f"{directory}/{image_basename}_distances.txt"
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
