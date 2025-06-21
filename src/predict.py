import cv2
import numpy as np
from tensorflow.keras.models import load_model
from preprocess import apply_filters
import logging
import os
from datetime import datetime


LOG_DIR = os.path.join('log', 'predict')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'predict.txt')),
        logging.StreamHandler()
    ]
)

class PneumoniaDetector:
    def __init__(self):
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.IMAGES_DIR = os.path.join(self.BASE_DIR, 'imagens')
        # self.MODEL_PATH = os.path.join(self.BASE_DIR, 'models', 'pneumonia_model.h5')
        self.MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'pneumonia_model.h5')
        self.model = None
        self.threshold = 0.65  # Threshold ajustado para melhor precisão
        self.load_model()

    def load_model(self):
        try:
            logging.info("Carregando modelo de pneumonia")
            self.model = load_model(self.MODEL_PATH)
            logging.info("Modelo carregado com sucesso")
        except Exception as e:
            logging.error(f"Falha ao carregar modelo: {str(e)}")
            raise

    def preprocess_image(self, img_path, img_size=(224, 224)):
        try:
            start_time = datetime.now()
            logging.info(f"Processando imagem: {img_path}")
            
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Falha ao carregar imagem {img_path}")
                
            img = apply_filters(img, img_size)
            img = img.reshape(1, img_size[0], img_size[1], 1)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            logging.info(f"Pré-processamento concluído em {elapsed:.2f}s")
            return img
            
        except Exception as e:
            logging.error(f"Erro no pré-processamento: {str(e)}")
            raise

    def predict(self, img_path):
        try:
            start_time = datetime.now()
            
            img = self.preprocess_image(img_path)
            
            logging.info("Fazendo predição...")
            prediction = self.model.predict(img, verbose=0)[0][0]
            
            # Suavização da predição
            if prediction > 0.85:
                result = 'PNEUMONIA (Alta probabilidade)'
            elif prediction > self.threshold:
                result = 'PNEUMONIA'
            elif prediction > 0.4:
                result = 'INDETERMINADO - Consulte um médico'
            else:
                result = 'NORMAL'
            
            elapsed = (datetime.now() - start_time).total_seconds()
            logging.info(f"Predição concluída em {elapsed:.2f}s - Resultado: {result} (Score: {prediction:.4f})")
            
            return result, float(prediction)
            
        except Exception as e:
            logging.error(f"Erro durante a predição: {str(e)}")
            return 'ERRO', 0.0

    def process_directory(self):
        if not os.path.exists(self.IMAGES_DIR):
            logging.error(f"Diretório de imagens não encontrado: {self.IMAGES_DIR}")
            return {}
        
        results = {}
        processed_count = 0
        
        logging.info(f"Iniciando processamento das imagens em: {self.IMAGES_DIR}")
        
        for root, _, files in os.walk(self.IMAGES_DIR):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(root, file)
                    try:
                        result, score = self.predict(img_path)
                        results[img_path] = (result, score)
                        processed_count += 1
                    except Exception as e:
                        logging.error(f"Falha ao processar {img_path}: {str(e)}")
                        results[img_path] = ('ERRO', 0.0)
        
        logging.info(f"Processamento concluído. {processed_count} imagens processadas.")
        return results

if __name__ == "__main__":
    detector = PneumoniaDetector()
    results = detector.process_directory()
    
    print("\n=== Resultados ===")
    for img_path, (result, score) in results.items():
        print(f"{os.path.basename(img_path)}: {result} (Score: {score:.4f})")
    
    print("\nLogs detalhados disponíveis em predict.log")