import os
import socket
import pickle
import cv2
import numpy as np
from tqdm import tqdm
import logging
import time
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from train_model import distributed_training

LOG_DIR = os.path.join('log', 'distributed_preprocess')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'distributed_preprocess.txt')),
        logging.StreamHandler()
    ]
)

class DistributedPreprocessor:
    def __init__(self, master_host: str = '127.0.0.1', master_port: int = 5000):
        self.master_host = master_host
        self.master_port = master_port
        self.chunk_size = 10
        self.max_retries = 3
        self.timeout = 30
        self.worker_cache = []
        self.cache_lock = threading.Lock()

    def load_image_paths(self, directory: str) -> List[Tuple[str, int]]:
        image_label_pairs = []
        classes = ['NORMAL', 'PNEUMONIA']

        for label in classes:
            class_dir = os.path.join(directory, label)
            if not os.path.exists(class_dir):
                logging.warning(f"Diretório {class_dir} não encontrado")
                continue

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(class_dir, img_name)
                    image_label_pairs.append((path, classes.index(label)))

        logging.info(f"Total de imagens encontradas: {len(image_label_pairs)}")
        return image_label_pairs

    def encode_image(self, path: str) -> bytes:
        img = cv2.imread(path)
        if img is None:
            logging.warning(f"Falha ao ler imagem: {path}")
            return None
        _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return buf.tobytes()

    def get_available_worker(self) -> Tuple[str, int]:
        with self.cache_lock:
            if not self.worker_cache:
                self._update_worker_cache()
            if not self.worker_cache:
                raise RuntimeError("Nenhum worker disponível")
            return self.worker_cache.pop(0)

    def _update_worker_cache(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)
                s.connect((self.master_host, self.master_port))
                s.sendall(b'GET_WORKERS')
                data = s.recv(4096)
                workers = pickle.loads(data)
                self.worker_cache = workers
                logging.info(f"Workers disponíveis atualizados: {len(workers)} workers")
        except Exception as e:
            logging.error(f"Falha ao obter workers do master: {str(e)}")
            raise

    def process_batch(self, image_batch: List[bytes]) -> List[np.ndarray]:
        for attempt in range(self.max_retries):
            worker = self.get_available_worker()
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(self.timeout)
                    s.connect(worker)
                    s.sendall(b'PROCESS_BATCH')

                    serialized = pickle.dumps(image_batch)
                    s.sendall(len(serialized).to_bytes(8, 'big'))
                    s.sendall(serialized)

                    size_data = s.recv(8)
                    size = int.from_bytes(size_data, 'big')
                    data = bytearray()
                    while len(data) < size:
                        packet = s.recv(min(4096, size - len(data)))
                        data.extend(packet)

                    return pickle.loads(data)

            except Exception as e:
                logging.warning(f"Tentativa {attempt + 1} falhou com worker {worker}: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(1)

    def distributed_preprocess(self, directory: str) -> Tuple[np.ndarray, np.ndarray]:
        image_label_pairs = self.load_image_paths(directory)

        # Encode as (encoded_image, label) tuples
        encoded_image_label_pairs = []
        for path, label in image_label_pairs:
            encoded = self.encode_image(path)
            if encoded is not None:
                encoded_image_label_pairs.append((encoded, label))

        if not encoded_image_label_pairs:
            raise ValueError("Nenhuma imagem foi codificada com sucesso.")

        # Split into chunks
        chunks = [encoded_image_label_pairs[i:i + self.chunk_size]
                  for i in range(0, len(encoded_image_label_pairs), self.chunk_size)]

        processed_images = []
        processed_labels = []

        progress = tqdm(total=len(encoded_image_label_pairs), desc="Processando imagens")

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(self.process_batch, [img for img, _ in chunk]): i
                for i, chunk in enumerate(chunks)
            }

            for future in as_completed(futures):
                idx = futures[future]
                chunk = chunks[idx]
                try:
                    results = future.result()
                    processed_images.extend(results)
                    processed_labels.extend([label for _, label in chunk])
                    progress.update(len(results))
                except Exception as e:
                    logging.error(f"Erro no processamento do chunk {idx}: {str(e)}")
                    progress.update(len(chunk))  # considera como "processado" mesmo com falha

        progress.close()

        if not processed_images:
            raise ValueError("Nenhuma imagem processada com sucesso.")

        try:
            images = np.array(processed_images).reshape(-1, 224, 224, 1)
        except Exception as e:
            logging.error(f"Erro ao converter imagens para array: {str(e)}")
            raise

        labels = np.array(processed_labels)

        logging.info(f"Total imagens processadas: {images.shape[0]}, Labels: {labels.shape[0]}")

        return images, labels

if __name__ == "__main__":
    preprocessor = DistributedPreprocessor()
    try:
        images, labels = preprocessor.distributed_preprocess('../data/train')
        logging.info(f"Processamento concluído. Imagens: {images.shape}, Labels: {labels.shape}")

        distributed_training(images, labels)
    except Exception as e:
        logging.error(f"Erro durante o pré-processamento ou treinamento: {str(e)}")
