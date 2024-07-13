from process_data import generate_train_data
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

def func_load_db(train_data_params, db_url):
    generate_train_data(**train_data_params)
    
    # db_url = "postgresql://postgres:postgres@192.168.4.23:5432/db"
    engine = create_engine(db_url)
    Base = declarative_base()

    class Measurement(Base):
        __tablename__ = 'measurements'
        id = Column(Integer, primary_key=True)
        type = Column(String)
        Testa_para_queixo = Column(Float)
        Ponta_do_nariz_para_testa = Column(Float)
        Ponta_do_nariz_para_queixo = Column(Float)
        Queixo_esquerdo_1_para_ponta_do_nariz = Column(Float)
        Queixo_direito_1_para_ponta_do_nariz = Column(Float)
        Queixo_esquerdo_2_para_ponta_do_nariz = Column(Float)
        Queixo_direito_2_para_ponta_do_nariz = Column(Float)
        Sombrancelha_maior_esquerda = Column(Float)
        Sombrancelha_menor_esquerda = Column(Float)
        sombrancelha_maior_direita = Column(Float)
        Sombrancelha_menor_direita = Column(Float)
        Largura_da_testa = Column(Float)
        Altura_da_testa = Column(Float)
        Distancia_entre_ponto_interno_de_cada_olho = Column(Float)
        Distancia_olho_esquerdo_para_ponta_do_nariz = Column(Float)
        Distancia_olho_direito_para_ponta_do_nariz = Column(Float)
        Distancia_entre_centro_de_cada_olho = Column(Float)
        Tamanho_olho_direito = Column(Float)
        Tamanho_olho_esquerdo = Column(Float)
        Centro_do_nariz_para_lado_esquerdo_do_nariz = Column(Float)
        Centro_do_nariz_para_lado_direito_do_nariz = Column(Float)
        Centro_labio_superior_para_ponta_do_nariz = Column(Float)
        Centro_labio_inferior_para_ponta_do_nariz = Column(Float)
        Ponto_labio_direito_para_ponta_do_nariz = Column(Float)
        Ponto_labio_esquerdo_para_ponta_do_nariz = Column(Float)
        Distancia_central_dos_labios = Column(Float)
        Ponto_externo_olho_direito_para_testa = Column(Float)
        Ponto_interno_olho_direito_para_testa = Column(Float)
        Ponto_externo_olho_esquerdo_para_testa = Column(Float)
        Ponto_interno_olho_esquerdo_para_testa = Column(Float)
        Ponto_externo_olho_direito_para_queixo = Column(Float)
        Ponto_interno_olho_direito_para_queixo = Column(Float)
        Ponto_externo_olho_esquerdo_para_queixo = Column(Float)
        Ponto_interno_olho_esquerdo_para_queixo = Column(Float)

    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    with open('/workdir/scripts/data/measurements_data.csv', 'r') as file:
        next(file)
        for line in file:
            data = line.strip().split(',')
            measurement_type = data[0]
            
            new_measurement = Measurement(
                type=measurement_type,
                Testa_para_queixo = float(data[1]),
                Ponta_do_nariz_para_testa = float(data[2]),
                Ponta_do_nariz_para_queixo = float(data[3]),
                Queixo_esquerdo_1_para_ponta_do_nariz = float(data[4]),
                Queixo_direito_1_para_ponta_do_nariz = float(data[5]),
                Queixo_esquerdo_2_para_ponta_do_nariz = float(data[6]),
                Queixo_direito_2_para_ponta_do_nariz = float(data[7]),
                Sombrancelha_maior_esquerda = float(data[8]),
                Sombrancelha_menor_esquerda = float(data[9]),
                sombrancelha_maior_direita = float(data[10]),
                Sombrancelha_menor_direita = float(data[11]),
                Largura_da_testa = float(data[12]),
                Altura_da_testa = float(data[13]),
                Distancia_entre_ponto_interno_de_cada_olho = float(data[14]),
                Distancia_olho_esquerdo_para_ponta_do_nariz = float(data[15]),
                Distancia_olho_direito_para_ponta_do_nariz = float(data[16]),
                Distancia_entre_centro_de_cada_olho = float(data[17]),
                Tamanho_olho_direito = float(data[18]),
                Tamanho_olho_esquerdo = float(data[19]),
                Centro_do_nariz_para_lado_esquerdo_do_nariz = float(data[20]),
                Centro_do_nariz_para_lado_direito_do_nariz = float(data[21]),
                Centro_labio_superior_para_ponta_do_nariz = float(data[22]),
                Centro_labio_inferior_para_ponta_do_nariz = float(data[23]),
                Ponto_labio_direito_para_ponta_do_nariz = float(data[24]),
                Ponto_labio_esquerdo_para_ponta_do_nariz = float(data[25]),
                Distancia_central_dos_labios = float(data[26]),
                Ponto_externo_olho_direito_para_testa = float(data[27]),
                Ponto_interno_olho_direito_para_testa = float(data[28]),
                Ponto_externo_olho_esquerdo_para_testa = float(data[29]),
                Ponto_interno_olho_esquerdo_para_testa = float(data[30]),
                Ponto_externo_olho_direito_para_queixo = float(data[31]),
                Ponto_interno_olho_direito_para_queixo = float(data[32]),
                Ponto_externo_olho_esquerdo_para_queixo = float(data[33]),
                Ponto_interno_olho_esquerdo_para_queixo = float(data[34]),
            )
            session.add(new_measurement)

    session.commit()
    session.close()


# train_data_params = {
#     'root_dir': '/workdir/webapp/static/images',
#     'config_path': '/workdir/scripts/config.json'
# }

# local_load_db(train_data_params)  