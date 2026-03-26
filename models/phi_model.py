import requests
from config import OPENROUTER_API_KEY, PHI_MODEL

class PhiModel:

    def generate(self, prompt):

        url = "https://openrouter.ai/api/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": PHI_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        response = requests.post(url, json=data, headers=headers)

        # Check status
        if response.status_code != 200:
            raise Exception(f"Phi-3 API returned status {response.status_code}: {response.text}")

        result = response.json()

        # Safe access
        try:
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            raise Exception(f"Unexpected Phi-3 response format: {result}")