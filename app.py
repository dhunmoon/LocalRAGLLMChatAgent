import os
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext, simpledialog, Toplevel, Label, Text, Button, END
from llm_engine import LLMEngine
from model_downloader import download_model, MODEL_PATH

class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Offline LLM Assistant")
        self.chat_history = []  # List of (role, message) tuples

        self.system_prompt = "You are a helpful assistant."  # Default system prompt

        # System prompt display at the top
        self.system_prompt_label = tk.Label(root, text=f"System Prompt: {self.system_prompt}", anchor='w', fg='blue', wraplength=900, justify='left')
        self.system_prompt_label.pack(fill=tk.X, padx=10, pady=(10, 0))

        # Wider chat area (width increased from 80 to 110)
        self.chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled', width=110, height=25)
        self.chat_area.pack(padx=10, pady=10)

        # Configure tags with colors
        self.chat_area.tag_config('system', foreground='blue')
        self.chat_area.tag_config('llm', foreground='green')
        self.chat_area.tag_config('user', foreground='orange')

        self.llm = None

        # Frame for entry + send button
        entry_frame = tk.Frame(root)
        entry_frame.pack(padx=10, pady=(0, 10), fill=tk.X)

        self.entry = tk.Text(entry_frame, height=4, width=90)  # Multi-line input box
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry.bind("<Shift-Return>", lambda e: self.insert_newline())
        self.entry.bind("<Return>", self.send_message)

        self.send_button = tk.Button(entry_frame, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.LEFT, padx=(5, 0))

        # Frame for file/folder/system prompt buttons aligned on one side (left)
        button_frame = tk.Frame(root)
        button_frame.pack(padx=10, pady=5, anchor='w')  # 'w' aligns to left

        self.file_button = tk.Button(button_frame, text="Select File", command=self.select_file)
        self.file_button.pack(side=tk.LEFT, padx=(0, 5))

        self.folder_button = tk.Button(button_frame, text="Select Folder", command=self.select_folder)
        self.folder_button.pack(side=tk.LEFT, padx=(0, 5))

        # System Prompt Button
        self.system_prompt_button = tk.Button(button_frame, text="System Prompt", command=self.show_system_prompt_popup)
        self.system_prompt_button.pack(side=tk.LEFT)

        self.append_chat("System", "Checking if model exists...")

        threading.Thread(target=self.check_and_prepare_model, daemon=True).start()

    def load_llm(self):
        try:
            self.llm = LLMEngine(system_prompt=self.system_prompt)
            # self.llm = LLMEngine(system_prompt=self.system_prompt)
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

    def build_prompt(self):
        prompt = f"<|system|>\n{self.system_prompt}\n"
        for role, message in self.chat_history:
            if role == "user":
                prompt += f"<|user|>\n{message}\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{message}\n"
        prompt += "<|assistant|>\n"
        return prompt

    def send_message(self, event=None):
        message = self.entry.get("1.0", tk.END).strip()
        if message:
            self.entry.delete("1.0", tk.END)
            self.append_chat("You", message)
            self.chat_history.append(("user", message))  # Add user message to history

            def process_response():
                if not self.llm:
                    self.append_chat("LLM", "LLM not loaded yet.")
                    return
                try:
                    prompt = self.build_prompt()
                    reply = self.llm.ask(prompt)
                    self.append_chat("LLM", reply)
                    self.chat_history.append(("assistant", reply))  # Add assistant reply to history
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
            self.append_chat("System", "Loading LLM...")
            threading.Thread(target=self.load_llm, daemon=True).start()
            self.append_chat("System", "Model already exists, ready to use.")
        else:
            self.append_chat("System", "Model not found, downloading now...")
            try:
                download_model()
                self.append_chat("System", "Model downloaded successfully.")
                threading.Thread(target=self.load_llm, daemon=True).start()
            except Exception as e:
                self.append_chat("Error", f"Failed to download model: {e}")

    def show_system_prompt_popup(self):
        popup = Toplevel(self.root)
        popup.title("Edit System Prompt")
        popup.transient(self.root)
        popup.grab_set()
        popup.geometry("600x200")

        label = Label(popup, text="Enter system prompt:")
        label.pack(pady=(10, 0))

        text_box = Text(popup, height=5, width=70)
        text_box.pack(padx=10, pady=10)
        text_box.insert(END, self.system_prompt)

        def save_prompt():
            self.system_prompt = text_box.get("1.0", END).strip()
            self.chat_history = []
            popup.destroy()
            self.system_prompt_label.config(text=f"System Prompt: {self.system_prompt}")
            self.llm = LLMEngine(system_prompt=self.system_prompt)
            
            # Reload LLM with new system prompt
            self.append_chat("System", "Reloading LLM with new system prompt...")
            # This is reloading the model withthe system proompt no need to load it again.
            # threading.Thread(target=self.load_llm, daemon=True).start()

        save_button = Button(popup, text="Save", command=save_prompt)
        save_button.pack(pady=(0, 10))

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()