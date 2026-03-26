# models/llama_model.py

import requests
import os


class LlamaModel:

    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")

        self.base_url = "https://openrouter.ai/api/v1"

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
    def generate(self, prompt):
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": "meta-llama/llama-3-70b-instruct",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        try:
            response = requests.post(url, headers=self.headers, json=payload)

            data = response.json()

            # 🔴 Handle API errors
            if "error" in data:
                raise Exception(f"API Error: {data['error']}")

            # 🔴 Validate structure
            if "choices" not in data:
                raise Exception(f"Invalid response format: {data}")

            return data["choices"][0]["message"]["content"]

        except Exception as e:
            print("LLaMA API failed:", e)
            raise e

    def embed(self, text):
        url = f"{self.base_url}/embeddings"

        payload = {
            "model": "text-embedding-3-small",
            "input": text
        }

        response = requests.post(url, headers=self.headers, json=payload)

        if response.status_code != 200:
            raise Exception(f"Embedding failed: {response.text}")

        return response.json()["data"][0]["embedding"]