import os
import pandas as pd
from image_processing import FaceMeshModel


def generate_train_data(root_dir, config_path):
    modelo = FaceMeshModel(config_path=config_path)
    
    i = 0

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.jpg'):
                image_path = os.path.join(subdir, file)
                modelo.process_image(image_path=image_path)

                modelo.calculate_and_draw_distance(10, 152, save=True, measure_type='Testa para queixo')
                modelo.calculate_and_draw_distance(1, 10, save=True, measure_type='Ponta do nariz para testa')
                modelo.calculate_and_draw_distance(1, 152, save=True, measure_type='Ponta do nariz para queixo')
                modelo.calculate_and_draw_distance(1, 361, save=True, measure_type='Queixo esquerdo 1 para ponta do nariz')
                modelo.calculate_and_draw_distance(1, 132, save=True, measure_type='Queixo direito 1 para ponta do nariz')
                modelo.calculate_and_draw_distance(1, 365, save=True, measure_type='Queixo esquerdo 2 para ponta do nariz')
                modelo.calculate_and_draw_distance(1, 150, save=True, measure_type='Queixo direito 2 para ponta do nariz')
                modelo.calculate_and_draw_distance(334, 443, save=True, measure_type='Sombrancelha maior esquerda')
                modelo.calculate_and_draw_distance(282, 334, save=True, measure_type='Sombrancelha menor esquerda')
                modelo.calculate_and_draw_distance(105, 223, save=True, measure_type='sombrancelha maior direita')
                modelo.calculate_and_draw_distance(52, 105, save=True, measure_type='Sombrancelha menor direita')
                modelo.calculate_and_draw_distance(54, 284, save=True, measure_type='Largura da testa')
                modelo.calculate_and_draw_distance(9, 10, save=True, measure_type='Altura da testa')
                modelo.calculate_and_draw_distance(133, 463, save=True, measure_type='Distancia entre ponto interno de cada olho')
                modelo.calculate_and_draw_distance(1, 159, save=True, measure_type='Distancia olho esquerdo para ponta do nariz')
                modelo.calculate_and_draw_distance(1, 386, save=True, measure_type='Distancia olho direito para ponta do nariz')
                modelo.calculate_and_draw_distance(159, 386, save=True, measure_type='Distancia entre centro de cada olho')
                modelo.calculate_and_draw_distance(133, 33, save=True, measure_type='Tamanho olho direito')
                modelo.calculate_and_draw_distance(263, 362, save=True, measure_type='Tamanho olho esquerdo')
                modelo.calculate_and_draw_distance(1, 278, save=True, measure_type='Centro do nariz para lado esquerdo do nariz')
                modelo.calculate_and_draw_distance(1, 48, save=True, measure_type='Centro do nariz para lado direito do nariz')
                modelo.calculate_and_draw_distance(0, 1, save=True, measure_type='Centro labio superior para ponta do nariz')
                modelo.calculate_and_draw_distance(1, 16, save=True, measure_type='Centro labio inferior para ponta do nariz')
                modelo.calculate_and_draw_distance(1, 61, save=True, measure_type='Ponto labio direito para ponta do nariz')
                modelo.calculate_and_draw_distance(1, 291, save=True, measure_type='Ponto labio esquerdo para ponta do nariz')
                modelo.calculate_and_draw_distance(0, 16, save=True, measure_type='Distancia central dos labios')
                modelo.calculate_and_draw_distance(10, 33, save=True, measure_type='Ponto externo olho direito para testa')
                modelo.calculate_and_draw_distance(10, 133, save=True, measure_type='Ponto interno olho direito para testa')
                modelo.calculate_and_draw_distance(10, 263, save=True, measure_type='Ponto externo olho esquerdo para testa')
                modelo.calculate_and_draw_distance(10, 362, save=True, measure_type='Ponto interno olho esquerdo para testa')
                modelo.calculate_and_draw_distance(33, 152, save=True, measure_type='Ponto externo olho direito para queixo')
                modelo.calculate_and_draw_distance(133, 152, save=True, measure_type='Ponto interno olho direito para queixo')
                modelo.calculate_and_draw_distance(152, 263, save=True, measure_type='Ponto externo olho esquerdo para queixo')
                modelo.calculate_and_draw_distance(152, 362, save=True, measure_type='Ponto interno olho esquerdo para queixo')

                i += 1

    print('Total de imagens processadas:', i)


    input_dir = '/workdir/scripts/data/specific_distances'
    data = []

    measurements = []

    # Geracao do CSV
    first_file = True
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(input_dir, filename)
            with open(filepath, 'r') as file:
                lines = file.readlines()
                if first_file:
                    for line in lines:
                        measure_name = line.strip().split()[-1]
                        if measure_name not in measurements:
                            measurements.append(measure_name)
                    first_file = False

    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            tipo = filename.split('_')[0]
            filepath = os.path.join(input_dir, filename)
            image_data = {'Type': tipo.capitalize()}
            
            for measure in measurements:
                image_data[measure] = 0

            with open(filepath, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    parts = line.strip().split()
                    measure_name = parts[-1]
                    distance = float(parts[-2])
                    image_data[measure_name] = distance

            data.append(image_data)

    df = pd.DataFrame(data)
    print(df)

    df = df[['Type'] + [col for col in df.columns if col != 'Type']]

    df.to_csv('/workdir/scripts/data/measurements_data.csv', index=False)
