from llama_cpp import Llama

class LLMEngine:
    def __init__(self, model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf", system_prompt="You are a helpful assistant."):
        self.model = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4  # or use os.cpu_count()
        )
        self.system_prompt = system_prompt

    def ask(self, prompt):
        output = self.model(
            prompt,
            max_tokens=256,
            temperature=0.7,
            stop=["<|user|>", "<|system|>"],
        )
        return output["choices"][0]["text"].strip()
