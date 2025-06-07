import os
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext
from llm_engine import LLMEngine
from model_downloader import download_model, MODEL_PATH

class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Offline LLM Assistant")


        # Wider chat area (width increased from 80 to 110)
        self.chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled', width=110, height=25)
        self.chat_area.pack(padx=10, pady=10)

        # Configure tags with colors
        self.chat_area.tag_config('system', foreground='blue')
        self.chat_area.tag_config('llm', foreground='green')
        self.chat_area.tag_config('user', foreground='orange')

        self.llm = None
        self.append_chat("System", "Loading LLM...")
        threading.Thread(target=self.load_llm, daemon=True).start()


        # Frame for entry + send button
        entry_frame = tk.Frame(root)
        entry_frame.pack(padx=10, pady=(0, 10), fill=tk.X)

        self.entry = tk.Text(entry_frame, height=4, width=90)  # Multi-line input box
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry.bind("<Shift-Return>", lambda e: self.insert_newline())
        self.entry.bind("<Return>", self.send_message)

        self.send_button = tk.Button(entry_frame, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.LEFT, padx=(5, 0))

        # Frame for file/folder buttons aligned on one side (left)
        button_frame = tk.Frame(root)
        button_frame.pack(padx=10, pady=5, anchor='w')  # 'w' aligns to left

        self.file_button = tk.Button(button_frame, text="Select File", command=self.select_file)
        self.file_button.pack(side=tk.LEFT, padx=(0, 5))

        self.folder_button = tk.Button(button_frame, text="Select Folder", command=self.select_folder)
        self.folder_button.pack(side=tk.LEFT)

        self.append_chat("System", "Checking if model exists...")

        threading.Thread(target=self.check_and_prepare_model, daemon=True).start()

    def load_llm(self):
        try:
            self.llm = LLMEngine()
            self.append_chat("System", "LLM loaded and ready.")
        except Exception as e:
            self.append_chat("System", f"Failed to load LLM: {e}")

    def append_chat(self, sender, message):
        self.chat_area.config(state='normal')
        if sender.lower() == "system":
            tag = 'system'
        elif sender.lower() == "llm":
            tag = 'llm'
        elif sender.lower() == "you":
            tag = 'user'
        else:
            tag = None

        if tag:
            self.chat_area.insert(tk.END, f"{sender}: {message}\n", tag)
        else:
            self.chat_area.insert(tk.END, f"{sender}: {message}\n")

        self.chat_area.config(state='disabled')
        self.chat_area.see(tk.END)

    def send_message(self, event=None):
        message = self.entry.get("1.0", tk.END).strip()
        if message:
            self.entry.delete("1.0", tk.END)
            self.append_chat("You", message)

            def process_response():
                if not self.llm:
                    self.append_chat("LLM", "LLM not loaded yet.")
                    return
                try:
                    reply = self.llm.ask(message)
                    self.append_chat("LLM", reply)
                except Exception as e:
                    self.append_chat("LLM", f"Error: {e}")

            threading.Thread(target=process_response, daemon=True).start()

        return "break"



    def select_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.append_chat("System", f"Selected file: {file_path}")

    def select_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.append_chat("System", f"Selected folder: {folder_path}")

    def insert_newline(self):
        self.entry.insert(tk.INSERT, "\n")
        return "break"



    def check_and_prepare_model(self):
        if os.path.exists(MODEL_PATH):
            self.append_chat("System", "Model already exists, ready to use.")
        else:
            self.append_chat("System", "Model not found, downloading now...")
            try:
                download_model()
                self.append_chat("System", "Model downloaded successfully.")
            except Exception as e:
                self.append_chat("Error", f"Failed to download model: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()
