from image_processing import FaceMeshModel

modelo = FaceMeshModel(config_path='config-example.json')

modelo.process_image(image_path='path_to_your_image.jpg')

modelo.calculate_and_draw_distance(0, 1, save=True, measure_type='specific distance example')

modelo.display_or_save_image()
