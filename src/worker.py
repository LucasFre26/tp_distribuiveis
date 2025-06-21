import os
import socket
import pickle
import cv2
import numpy as np
from preprocess import apply_filters
import logging
from datetime import datetime
import threading
import time
from typing import Tuple
from tensorflow.keras.regularizers import l2  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryFocalCrossentropy


LOG_DIR = os.path.join('log', 'worker')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'worker.txt')),
        logging.StreamHandler()
    ]
)

class WorkerNode:
    def __init__(self, master_host: str = '127.0.0.1', master_port: int = 5000):
        self.master_host = master_host
        self.master_port = master_port
        self.host = '0.0.0.0'
        self.port = 0
        self.max_data_size = 10 * 1024 * 1024
        self.worker_socket = None
        self.running = False
        self.heartbeat_interval = 30

    def start(self):
        test_img = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        apply_filters(test_img)

        self.worker_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.worker_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.worker_socket.bind((self.host, self.port))
        self.worker_socket.listen()
        self.port = self.worker_socket.getsockname()[1]

        self._register_with_master()

        threading.Thread(target=self._send_heartbeat, daemon=True).start()

        self.running = True
        logging.info(f"Worker iniciado em {self.host}:{self.port}")

        try:
            while self.running:
                conn, addr = self.worker_socket.accept()
                threading.Thread(target=self._handle_connection, args=(conn, addr), daemon=True).start()
        except KeyboardInterrupt:
            self.stop()
        except Exception as e:
            logging.error(f"Erro no worker: {str(e)}")
            self.stop()

    def stop(self):
        self.running = False
        if self.worker_socket:
            self.worker_socket.close()
        logging.info("Worker encerrado")

    def _register_with_master(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.master_host, self.master_port))
                s.sendall(b'REGISTER')

                ip = socket.gethostbyname(socket.gethostname())
                data = pickle.dumps((ip, self.port))

                s.sendall(len(data).to_bytes(8, 'big'))
                s.sendall(data)

                response = s.recv(1024)
                if response != b'OK':
                    raise RuntimeError("Falha no registro com o master")
                logging.info(f"Registrado no master em {self.master_host}:{self.master_port}")
        except Exception as e:
            logging.error(f"Falha ao registrar no master: {str(e)}")
            raise

    def _send_heartbeat(self):
        while self.running:
            time.sleep(self.heartbeat_interval)
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(5)
                    s.connect((self.master_host, self.master_port))
                    s.sendall(b'HEARTBEAT')
                    response = s.recv(1024)
                    if response != b'ALIVE':
                        logging.warning("Master não respondeu ao heartbeat")
            except Exception as e:
                logging.error(f"Falha no heartbeat: {str(e)}")

    def _handle_connection(self, conn: socket.socket, addr: Tuple[str, int]):
        try:
            with conn:
                conn.settimeout(60)
                operation = conn.recv(1024)

                if operation == b'PROCESS_BATCH':
                    self._handle_batch_processing(conn)
                elif operation == b'PROCESS_SINGLE':
                    self._handle_single_processing(conn)
                elif operation == b'HEARTBEAT':
                    conn.sendall(b'ALIVE')
                elif operation == b'TRAIN_SHARD':
                    self._handle_train_shard(conn)
                else:
                    raise ValueError("Operação desconhecida")
        except Exception as e:
            logging.error(f"Erro na conexão com {addr}: {str(e)}")

    def _handle_batch_processing(self, conn: socket.socket):
        try:
            size = int.from_bytes(self._recv_exact(conn, 8), 'big')
            if size <= 0 or size > self.max_data_size:
                raise ValueError(f"Tamanho de dados inválido: {size}")

            data = self._recv_exact(conn, size)
            image_batch = pickle.loads(data)

            results = [self._process_single_image(img_data) for img_data in image_batch]

            response = pickle.dumps(results)
            conn.sendall(len(response).to_bytes(8, 'big'))
            conn.sendall(response)
        except Exception as e:
            logging.error(f"Erro no processamento em lote: {str(e)}")
            conn.sendall((0).to_bytes(8, 'big'))

    def _handle_single_processing(self, conn: socket.socket):
        try:
            size = int.from_bytes(self._recv_exact(conn, 8), 'big')
            if size <= 0 or size > self.max_data_size:
                raise ValueError(f"Tamanho de dados inválido: {size}")

            data = self._recv_exact(conn, size)
            image_data = pickle.loads(data)[0]

            result = self._process_single_image(image_data)

            conn.sendall(b'\x01')
            response = pickle.dumps([result] if result is not None else [None])
            conn.sendall(len(response).to_bytes(8, 'big'))
            conn.sendall(response)
        except Exception as e:
            logging.error(f"Erro no processamento único: {str(e)}")
            conn.sendall(b'\x00')
            conn.sendall((0).to_bytes(8, 'big'))

    def _handle_train_shard(self, conn: socket.socket):
        try:
            # Recebe os dados
            size = int.from_bytes(self._recv_exact(conn, 8), 'big')
            data = self._recv_exact(conn, size)
            X_shard, y_shard, global_weights = pickle.loads(data)

            # Cria modelo local
            local_model = self.build_model()
            
            # Verifica compatibilidade dos pesos
            if len(global_weights) != len(local_model.get_weights()):
                raise ValueError(f"Incompatibilidade de pesos. Esperado: {len(local_model.get_weights())}, Recebido: {len(global_weights)}")
            
            # Seta pesos
            local_model.set_weights(global_weights)
            
            # Treina
            local_model.fit(X_shard, y_shard, epochs=1, verbose=0)
            
            # Retorna novos pesos
            response = pickle.dumps(local_model.get_weights())
            conn.sendall(len(response).to_bytes(8, 'big'))
            conn.sendall(response)
            
        except Exception as e:
            logging.error(f"Erro no treinamento: {str(e)}")
            conn.sendall((0).to_bytes(8, 'big'))

    def _process_single_image(self, img_data, img_size=(224, 224)):
        try:
            start_time = datetime.now()
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None:
                logging.warning("Falha ao decodificar imagem")
                return None
            filtered = apply_filters(img, img_size)
            elapsed = (datetime.now() - start_time).total_seconds()
            logging.info(f"Imagem processada em {elapsed:.2f}s")
            return filtered
        except Exception as e:
            logging.error(f"Erro ao processar imagem: {str(e)}")
            return None

    @staticmethod
    def build_model(input_shape=(224,224,1)):
        model = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=input_shape, padding='same'),
            BatchNormalization(),
            MaxPooling2D(2,2),
            Dropout(0.2),
            
            Conv2D(64, (3,3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(2,2),
            Dropout(0.3),
            
            Conv2D(128, (3,3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(2,2),
            Dropout(0.4),
            
            Flatten(),
            
            Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.0001),
                    loss=BinaryFocalCrossentropy(gamma=2.0),
                    metrics=['accuracy'])
        
        return model

    def _recv_exact(self, conn: socket.socket, num_bytes: int) -> bytes:
        data = bytearray()
        while len(data) < num_bytes:
            packet = conn.recv(min(4096, num_bytes - len(data)))
            if not packet:
                raise ConnectionError("Conexão interrompida durante recv")
            data.extend(packet)
        return bytes(data)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--master-host', default='127.0.0.1', help='Endereço do master')
    parser.add_argument('--master-port', type=int, default=5000, help='Porta do master')
    args = parser.parse_args()

    worker = WorkerNode(master_host=args.master_host, master_port=args.master_port)
    try:
        worker.start()
    except KeyboardInterrupt:
        worker.stop()
    except Exception as e:
        logging.error(f"Erro fatal no worker: {str(e)}")
        worker.stop()