import os
import pandas as pd
from sqlalchemy import create_engine, Table, Column, Float, Integer, String, MetaData, insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import inspect

DATABASE_TYPE = 'postgresql'
DBAPI = 'psycopg2'
ENDPOINT = os.environ.get('DATABASE_HOST', '192.168.4.23')
USER = os.environ.get('DATABASE_USER', 'postgres')
PASSWORD = os.environ.get('DATABASE_PASSWORD', 'postgres')
PORT = os.environ.get('DATABASE_PORT', 5432)
DATABASE = os.environ.get('DATABASE_NAME', 'db')

engine = create_engine(f'postgresql://{USER}:{PASSWORD}@{ENDPOINT}:{PORT}/{DATABASE}')

path_to_txt_files = "/workdir/scripts/data/specific_distances"

def create_table_if_not_exists(engine):
    print('CRIANDO BANCO')
    metadata = MetaData()
    table = Table('measurements', metadata,
                  Column('id', Integer, primary_key=True),
                  Column('type', String),
                  Column('Testa_para_queixo', Float),
                  Column('Ponta_do_nariz_para_testa', Float),
                  Column('Ponta_do_nariz_para_queixo', Float),
                  Column('Queixo_esquerdo_1_para_ponta_do_nariz', Float),
                  Column('Queixo_direito_1_para_ponta_do_nariz', Float),
                  Column('Queixo_esquerdo_2_para_ponta_do_nariz', Float),
                  Column('Queixo_direito_2_para_ponta_do_nariz', Float),
                  Column('Sombrancelha_maior_esquerda', Float),
                  Column('Sombrancelha_menor_esquerda', Float),
                  Column('sombrancelha_maior_direita', Float),
                  Column('Sombrancelha_menor_direita', Float),
                  Column('Largura_da_testa', Float),
                  Column('Altura_da_testa', Float),
                  Column('Distancia_entre_ponto_interno_de_cada_olho', Float),
                  Column('Distancia_olho_esquerdo_para_ponta_do_nariz', Float),
                  Column('Distancia_olho_direito_para_ponta_do_nariz', Float),
                  Column('Distancia_entre_centro_de_cada_olho', Float),
                  Column('Tamanho_olho_direito', Float),
                  Column('Tamanho_olho_esquerdo', Float),
                  Column('Centro_do_nariz_para_lado_esquerdo_do_nariz', Float),
                  Column('Centro_do_nariz_para_lado_direito_do_nariz', Float),
                  Column('Centro_labio_superior_para_ponta_do_nariz', Float),
                  Column('Centro_labio_inferior_para_ponta_do_nariz', Float),
                  Column('Ponto_labio_direito_para_ponta_do_nariz', Float),
                  Column('Ponto_labio_esquerdo_para_ponta_do_nariz', Float),
                  Column('Distancia_central_dos_labios', Float),
                  Column('Ponto_externo_olho_direito_para_testa', Float),
                  Column('Ponto_interno_olho_direito_para_testa', Float),
                  Column('Ponto_externo_olho_esquerdo_para_testa', Float),
                  Column('Ponto_interno_olho_esquerdo_para_testa', Float),
                  Column('Ponto_externo_olho_direito_para_queixo', Float),
                  Column('Ponto_interno_olho_direito_para_queixo', Float),
                  Column('Ponto_externo_olho_esquerdo_para_queixo', Float),
                  Column('Ponto_interno_olho_esquerdo_para_queixo', Float),
                 )
    inspector = inspect(engine)
    if not inspector.has_table('measurements', schema=None):
        metadata.create_all(engine)
    
    return table

def process_txt_file(file_path):
    data = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(' ')
                measure_name = '_'.join(parts[3:])
                distance = float(parts[2])
                data[measure_name] = distance
    except Exception as e:
        print(f"Erro ao processar arquivo {file_path}: {e}")
    return data

def load_data_to_db(table, engine, data):
    try:
        with engine.connect() as connection:
            stmt = insert(table).values(data)
            connection.execute(stmt)
            print("Dados inseridos com sucesso.")
    except SQLAlchemyError as e:
        print(f"Erro ao inserir dados no banco: {e}")

def check_database():
    """ Verifica se o banco de dados está vazio e inicializa se necessário. """
    inspector = inspect(engine)
    if not inspector.get_table_names():
        print("Banco de dados está vazio, executando local_load_db.py")
        os.system("python local_load_db.py")
    else:
        print("Banco de dados já contém dados.")

def main():
    table = create_table_if_not_exists(engine)
    check_database()
    for filename in os.listdir(path_to_txt_files):
        if filename.endswith(".txt"):
            file_path = os.path.join(path_to_txt_files, filename)
            print(f"Processando arquivo: {file_path}")
            data = process_txt_file(file_path)
            load_data_to_db(table, engine, data)
