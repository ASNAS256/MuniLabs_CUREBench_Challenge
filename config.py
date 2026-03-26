import os

# API KEYS
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# MODEL NAMES (in use)
DEEPSEEK_MODEL = "deepseek-reasoner"

# MODEL NAMES (not in use yet)
LLAMA_MODEL = "meta-llama/llama-3-70b-instruct"
PHI_MODEL = "microsoft/phi-3-mini-instruct"  # Confirm actual endpoint

# ROUTING SETTINGS
USE_MULTI_MODEL = False

GEMINI_MODEL = "gemini-2.0-flash"   # ✅ correct

API_KEY = os.getenv("GOOGLE_API_KEY")

TEMPERATURE = 0
MAX_TOKENS = 128