import socket
import pickle
import threading
import logging
from queue import Queue
from typing import List, Tuple, Dict
import os
import time


LOG_DIR = os.path.join('log', 'master')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'master.txt')),
        logging.StreamHandler()
    ]
)

class WorkerManager:
    def __init__(self):
        self.workers: List[Tuple[str, int]] = []
        self.worker_status: Dict[Tuple[str, int], bool] = {}
        self.worker_lock = threading.Lock()
        self.heartbeat_interval = 30
        self.task_queue = Queue()

    def add_worker(self, ip: str, port: int):
        with self.worker_lock:
            worker = (ip, port)
            if worker not in self.workers:
                self.workers.append(worker)
                self.worker_status[worker] = True
                logging.info(f"Worker adicionado: {ip}:{port}")

    def remove_worker(self, ip: str, port: int):
        with self.worker_lock:
            worker = (ip, port)
            if worker in self.workers:
                self.workers.remove(worker)
                self.worker_status.pop(worker, None)
                logging.info(f"Worker removido: {ip}:{port}")

    def update_worker_status(self, ip: str, port: int, status: bool):
        with self.worker_lock:
            worker = (ip, port)
            if worker in self.worker_status:
                self.worker_status[worker] = status

    def get_available_workers(self) -> List[Tuple[str, int]]:
        with self.worker_lock:
            return [worker for worker, status in self.worker_status.items() if status]

    def start_heartbeat_checker(self):
        def checker():
            while True:
                time.sleep(self.heartbeat_interval)
                self.check_worker_heartbeats()
        threading.Thread(target=checker, daemon=True).start()

    def check_worker_heartbeats(self):
        with self.worker_lock:
            for worker in list(self.worker_status.keys()):
                ip, port = worker
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.settimeout(5)
                        s.connect((ip, port))
                        s.sendall(b'HEARTBEAT')
                        response = s.recv(1024)
                        if response == b'ALIVE':
                            self.worker_status[worker] = True
                        else:
                            self.worker_status[worker] = False
                            logging.warning(f"Worker {ip}:{port} não respondeu corretamente")
                except Exception as e:
                    self.worker_status[worker] = False
                    logging.warning(f"Falha no heartbeat do worker {ip}:{port}: {str(e)}")

    def start_task_dispatcher(self):
        def dispatcher():
            while True:
                task = self.task_queue.get()
                workers = self.get_available_workers()
                if workers:
                    worker = workers[0]  # Implementar lógica melhor em produção
                    self.send_task_to_worker(worker, task)
        threading.Thread(target=dispatcher, daemon=True).start()

    def send_task_to_worker(self, worker, task):
        try:
            ip, port = worker
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(30)
                s.connect((ip, port))
                s.sendall(task['type'].encode())
                
                serialized = pickle.dumps(task['data'])
                s.sendall(len(serialized).to_bytes(8, 'big'))
                s.sendall(serialized)
                
                size_data = s.recv(8)
                size = int.from_bytes(size_data, 'big')
                response = bytearray()
                while len(response) < size:
                    packet = s.recv(min(4096, size - len(response)))
                    response.extend(packet)
                
                return pickle.loads(response)
        except Exception as e:
            logging.error(f"Erro ao enviar tarefa para {worker}: {str(e)}")
            return None

class MasterServer:
    def __init__(self, host: str = '0.0.0.0', port: int = 5000):
        self.host = host
        self.port = port
        self.worker_manager = WorkerManager()
        self.task_queue = Queue()

    def start(self):
        self.worker_manager.start_heartbeat_checker()
        self.worker_manager.start_task_dispatcher()
        threading.Thread(target=self._accept_workers, daemon=True).start()
        logging.info(f"Master server iniciado em {self.host}:{self.port}")

    def _accept_workers(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen()

            while True:
                conn, addr = s.accept()
                threading.Thread(target=self._handle_worker_connection, args=(conn, addr), daemon=True).start()

    def _handle_worker_connection(self, conn: socket.socket, addr: Tuple[str, int]):
        try:
            with conn:
                data = conn.recv(1024)
                if data == b'REGISTER':
                    size_data = conn.recv(8)
                    size = int.from_bytes(size_data, 'big')
                    serialized = self._recv_exact(conn, size)
                    ip, port = pickle.loads(serialized)
                    self.worker_manager.add_worker(ip, port)
                    conn.sendall(b'OK')
                elif data == b'HEARTBEAT':
                    conn.sendall(b'ALIVE')
                elif data == b'GET_WORKERS':
                    with self.worker_manager.worker_lock:
                        workers = [(ip, port) for (ip, port), status in self.worker_manager.worker_status.items() if status]
                    serialized = pickle.dumps(workers)
                    conn.sendall(serialized)
                elif data == b'TRAIN_SHARD':
                    size = int.from_bytes(conn.recv(8), 'big')
                    task_data = pickle.loads(self._recv_exact(conn, size))
                    task = {'type': 'TRAIN_SHARD', 'data': task_data}
                    self.worker_manager.task_queue.put(task)
        except Exception as e:
            logging.error(f"Erro na conexão com worker {addr}: {str(e)}")

    def _recv_exact(self, conn: socket.socket, num_bytes: int) -> bytes:
        data = bytearray()
        while len(data) < num_bytes:
            packet = conn.recv(min(4096, num_bytes - len(data)))
            if not packet:
                raise ConnectionError("Conexão interrompida durante recv")
            data.extend(packet)
        return bytes(data)

if __name__ == "__main__":
    master = MasterServer()
    master.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Master server encerrado")