from sqlalchemy import create_engine, Table, Column, Float, Integer, String, MetaData, insert
from sqlalchemy_utils import database_exists, create_database
from sqlalchemy.ext.declarative import declarative_base
from process_data import generate_train_data
from sqlalchemy.orm import sessionmaker

def create_database_if_not_exists(url):
    engine = create_engine(url)
    if not database_exists(engine.url):
        create_database(engine.url)
        print("Banco de dados criado.")
    
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
    
    if not engine.dialect.has_table(engine.connect(), 'measurements'):
        metadata.create_all(engine)
        print("Tabela criada.")
    
    return engine, table
