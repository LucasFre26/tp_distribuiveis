import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from predict import PneumoniaDetector  # Importar a classe correta

# Criar uma instância do detector
detector = PneumoniaDetector()

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Carregar e exibir a imagem
        img = Image.open(file_path).resize((224, 224))
        img_tk = ImageTk.PhotoImage(img)
        panel.configure(image=img_tk)
        panel.image = img_tk
        
        # Fazer a predição
        result, confidence = detector.predict(file_path)
        label_result.config(text=f'Result: {result} (Confidence: {confidence:.2f})')

# Configuração da GUI
root = tk.Tk()
root.title('Pneumonia Detection')

panel = tk.Label(root)
panel.pack()

btn_upload = tk.Button(root, text='Upload Image', command=upload_image)
btn_upload.pack()

label_result = tk.Label(root, text='Result: ')
label_result.pack()

root.mainloop()