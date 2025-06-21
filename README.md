# 🔬 Pneumonia Detection com Computação Distribuída

Este projeto implementa um sistema completo de **classificação de pneumonia em radiografias de tórax**, utilizando **computação distribuída** para acelerar o pré-processamento e o treinamento de modelos de deep learning. A arquitetura segue o modelo **Master–Worker**, com suporte a **treinamento federado**, **processamento paralelo**, e **interface gráfica para predição**.

---

## 📁 Estrutura do Projeto

```

.
├── gui.py                      # Interface gráfica para predição
├── predict.py                 # Predição em imagens usando o modelo treinado
├── train\_model.py            # Treinamento federado do modelo
├── distributed\_preprocess.py # Pré-processamento paralelo distribuído
├── master.py                 # Servidor mestre (coordena os workers)
├── worker.py                 # Nó worker (executa processamento ou treino)
├── preprocess.py             # Filtros e normalização de imagem
├── models/                   # Onde o modelo treinado é salvo
└── log/                      # Logs separados por componente

````

---

## 🧠 Conceitos de Computação Distribuída Abordados

- Arquitetura **Master–Worker**
- **Treinamento federado** (os dados não são compartilhados, apenas os pesos)
- **Balanceamento de carga**
- **Heartbeats** e tolerância a falhas
- **Processamento paralelo** com ThreadPoolExecutor
- Comunicação entre processos com **Sockets TCP**
- Logs e monitoramento distribuído

---

## 🧪 Requisitos

- Python 3.8+
- TensorFlow
- OpenCV
- NumPy
- Scikit-learn
- Pillow (para GUI)
- tqdm

```bash
pip install -r requirements.txt
````

---

## 📂 Estrutura esperada de dados

```bash
data/
└── train/
    ├── NORMAL/
    │   ├── img1.jpg
    │   └── ...
    └── PNEUMONIA/
        ├── img2.jpg
        └── ...
```

---

## 🚀 Como Executar

### 1. Inicie o Master

```bash
python master.py
```

### 2. Inicie 1 ou mais Workers (em terminais diferentes)

```bash
python worker.py
```

### 3. Faça o pré-processamento distribuído e treine o modelo

```bash
python distributed_preprocess.py
```

### 4. Realize predições (modo terminal)

```bash
python predict.py
```

### 5. Ou use a interface gráfica

```bash
python gui.py
```

---

## 📊 Resultados

O modelo utiliza:

* Rede CNN com 3 blocos convolucionais
* `BatchNormalization`, `Dropout`, `L2 regularization`
* Função de perda `BinaryFocalCrossentropy` para lidar com desbalanceamento
* `EarlyStopping` para evitar overfitting

---

## 📌 Logs

Todos os logs são salvos em subpastas dentro da pasta `log/`, por exemplo:

```bash
log/
├── master/
├── worker/
├── train/
├── predict/
└── preprocess/
```

---

##  Autores

**Lucas Jose de Freitas, **
**Isaac Alves Schuenck, **
**Gabriel Avelar Sabato**
---

