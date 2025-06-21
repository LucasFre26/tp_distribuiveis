import os
import numpy as np
import logging
import socket
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryFocalCrossentropy
from tensorflow.keras.regularizers import l2  # Importação adicionada aqui
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

LOG_DIR = os.path.join('log', 'train')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'train.txt')),
        logging.StreamHandler()
    ]
)

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

def distributed_training(X, y, master_host='127.0.0.1', master_port=5000, epochs=20, batch_size=32):
    # Verificação e divisão dos dados
    assert len(X) == len(y), f"Tamanhos inconsistentes: X={len(X)}, y={len(y)}"
    
    # Divisão estratificada para manter proporção de classes
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')
    
    # Balanceamento de classes
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}
    
    # Obter workers
    workers = get_available_workers(master_host, master_port)
    if not workers:
        raise RuntimeError("Nenhum worker disponível para treinamento")
    
    num_workers = len(workers)
    shard_size = len(X_train) // num_workers
    
    # Inicializar modelo global
    global_model = build_model(input_shape=X_train.shape[1:])
    global_weights = global_model.get_weights()
    
    # Callback para early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1)
    
    # Loop de treinamento federado
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        logging.info(f"Iniciando época {epoch+1}/{epochs}")
        new_weights = []
        
        # Enviar shards para workers com balanceamento de classes
        for idx, worker in enumerate(workers):
            start_idx = idx * shard_size
            end_idx = (idx + 1) * shard_size if idx < num_workers - 1 else len(X_train)
            
            X_shard = X_train[start_idx:end_idx]
            y_shard = y_train[start_idx:end_idx]
            
            # Listas para armazenar os dados aumentados
            augmented_images = []
            augmented_labels = []
            
            for x, y in zip(X_shard, y_shard):
                # Adiciona a imagem original
                augmented_images.append(x)
                augmented_labels.append(y)
                
                if y == 1:  # Aumenta mais exemplos da classe minoritária
                    for _ in range(2):
                        # Garante que a imagem tem shape (224,224,1) antes da transformação
                        x_reshaped = x.reshape(224, 224, 1)
                        augmented_x = train_datagen.random_transform(x_reshaped)
                        # Remove dimensão extra se necessário e adiciona à lista
                        augmented_images.append(augmented_x.reshape(224, 224))
                        augmented_labels.append(y)
            
            # Converte para numpy array garantindo o shape correto
            X_augmented = np.stack([img.reshape(224, 224) for img in augmented_images])
            y_augmented = np.array(augmented_labels)
            
            # Verifica os shapes antes de enviar
            logging.debug(f"Shape X_augmented: {X_augmented.shape}, y_augmented: {y_augmented.shape}")
            
            # Enviar para worker
            updated_weights = send_train_shard(
                worker, 
                X_augmented,
                y_augmented,
                global_weights)
            
            if updated_weights is not None:
                new_weights.append(updated_weights)
        
        # Média federada dos pesos
        if new_weights:
            global_weights = [np.mean(layer_weights, axis=0) for layer_weights in zip(*new_weights)]
            global_model.set_weights(global_weights)
            
            # Avaliação do modelo
            val_loss, val_acc = global_model.evaluate(X_val, y_val, verbose=0)
            logging.info(f"Época {epoch+1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping.patience:
                    logging.info("Early stopping triggered")
                    break
    
    # Salvar modelo final
    model_dir = os.path.join('models')
    os.makedirs(model_dir, exist_ok=True)
    global_model.save(os.path.join(model_dir, 'pneumonia_model.h5'))
    logging.info("Modelo salvo com sucesso.")

def get_available_workers(master_host, master_port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(5)
            s.connect((master_host, master_port))
            s.sendall(b'GET_WORKERS')
            data = s.recv(4096)
            return pickle.loads(data)
    except Exception as e:
        logging.error(f"Falha ao obter workers: {str(e)}")
        return []

def send_train_shard(worker, X_shard, y_shard, model_weights):
    try:
        # Cria modelo temporário para verificação
        temp_model = build_model()
        expected_length = len(temp_model.get_weights())
        
        if len(model_weights) != expected_length:
            raise ValueError(f"Pesos incompatíveis. Esperado: {expected_length}, Tem: {len(model_weights)}")
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(120)  # Aumente o timeout para 2 minutos
            s.connect(worker)
            
            # Envia o comando
            s.sendall(b'TRAIN_SHARD')
            
            # Prepara os dados
            payload = pickle.dumps((X_shard, y_shard, model_weights))
            
            # Envia o tamanho primeiro
            s.sendall(len(payload).to_bytes(8, 'big'))
            
            # Envia os dados em chunks
            chunk_size = 4096
            for i in range(0, len(payload), chunk_size):
                s.sendall(payload[i:i+chunk_size])
            
            # Recebe a resposta
            size_data = s.recv(8)
            if not size_data:
                raise ValueError("No size data received")
                
            size = int.from_bytes(size_data, 'big')
            data = bytearray()
            
            while len(data) < size:
                remaining = size - len(data)
                packet = s.recv(min(4096, remaining))
                if not packet:
                    raise ConnectionError("Connection broken during data receive")
                data.extend(packet)
                
            return pickle.loads(data)
            
    except Exception as e:
        logging.error(f"Erro ao enviar shard para {worker}: {str(e)}")
        return None