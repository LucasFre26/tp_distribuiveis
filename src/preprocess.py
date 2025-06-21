import os
import cv2
import numpy as np
import logging


LOG_DIR = os.path.join('log', 'preprocess')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'preprocess.txt')),
        logging.StreamHandler()
    ]
)

def apply_filters(img, img_size=(224, 224)):
    try:
        logging.info("Aplicando filtros na imagem")
        
        # Redimensiona com interpolação de alta qualidade
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        logging.debug("Redimensionamento concluído")
        
        # Equalização de histograma para melhorar contraste
        img = cv2.equalizeHist(img)
        
        # Normalização local para lidar com variações de iluminação
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        
        # Filtro Gaussiano para reduzir ruído
        img = cv2.GaussianBlur(img, (3,3), 0)
        
        # Filtro Sobel com pesos ajustados
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = cv2.magnitude(sobelx, sobely)
        sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # Filtro Laplace com ajuste de escala
        laplace = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
        laplace = cv2.normalize(laplace, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # Combinação ponderada dos filtros
        combined = (0.4 * sobel_combined.astype(np.float32) +
                    0.3 * laplace.astype(np.float32) +
                    0.3 * img.astype(np.float32))
        
        # Normalização final
        combined = combined / 255.0
        logging.info("Filtros aplicados com sucesso")
        
        return combined
        
    except Exception as e:
        logging.error(f"Erro ao aplicar filtros: {str(e)}")
        raise