from google import genai
import os

class GeminiModel:

    def __init__(self, model_name):
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model_name = model_name

    def generate(self, prompt):

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )

        return response.text