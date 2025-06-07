import os
import requests

MODEL_DIR = "models"
MODEL_FILENAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
MODEL_URL = (
    "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/"
    + MODEL_FILENAME
)

def download_model(progress_callback=None, complete_callback=None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        try:
            with requests.get(MODEL_URL, stream=True) as response:
                response.raise_for_status()
                total = int(response.headers.get('content-length', 0))
                downloaded = 0
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            percent = int((downloaded / total) * 100)
                            if progress_callback:
                                progress_callback(percent)
            if complete_callback:
                complete_callback()
        except Exception as e:
            if progress_callback:
                progress_callback(-1)
            print("Download failed:", e)
    else:
        if complete_callback:
            complete_callback()
