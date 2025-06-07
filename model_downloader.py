import os
import requests
from tkinter import messagebox

MODEL_DIR = "models"
MODEL_FILENAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
MODEL_URL = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print("Model downloaded successfully.")
        except requests.exceptions.RequestException as e:
            messagebox.showerror("Download Error", f"An error occurred while downloading the model: {e}")
    else:
        print("Model already exists.")
