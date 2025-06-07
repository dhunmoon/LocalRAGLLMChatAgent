import tkinter as tk
from tkinter import filedialog, scrolledtext

class LLMChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Offline LLM Assistant")

        # Chat display
        self.chat_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=25, state='disabled')
        self.chat_display.grid(row=0, column=0, columnspan=3, padx=10, pady=10)

        # User input
        self.input_text = tk.StringVar()
        self.input_entry = tk.Entry(root, textvariable=self.input_text, width=60)
        self.input_entry.grid(row=1, column=0, padx=10, pady=5)
        self.input_entry.bind("<Return>", self.handle_send)

        # Send button
        self.send_button = tk.Button(root, text="Send", command=self.handle_send)
        self.send_button.grid(row=1, column=1, padx=5, pady=5)

        # Folder select button
        self.folder_button = tk.Button(root, text="Select Folder", command=self.select_folder)
        self.folder_button.grid(row=2, column=0, padx=10, pady=5, sticky="w")

        # File select button
        self.file_button = tk.Button(root, text="Select File", command=self.select_file)
        self.file_button.grid(row=2, column=1, padx=5, pady=5, sticky="w")

    def append_chat(self, sender, message):
        self.chat_display.configure(state='normal')
        self.chat_display.insert(tk.END, f"{sender}: {message}\n\n")
        self.chat_display.configure(state='disabled')
        self.chat_display.see(tk.END)

    def handle_send(self, event=None):
        user_message = self.input_text.get().strip()
        if user_message:
            self.append_chat("You", user_message)
            self.input_text.set("")  # Clear input
            self.root.after(100, self.fake_llm_response, user_message)  # Placeholder response

    def fake_llm_response(self, user_message):
        # TODO: Replace with actual LLM logic
        response = f"(LLM Response to: {user_message})"
        self.append_chat("Assistant", response)

    def select_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.append_chat("System", f"Selected folder: {folder}")

    def select_file(self):
        file = filedialog.askopenfilename()
        if file:
            self.append_chat("System", f"Selected file: {file}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LLMChatApp(root)
    root.mainloop()
