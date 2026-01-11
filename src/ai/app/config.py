import os

# Database Path
DB_PATH = os.getenv("DB_PATH", "flaicheckbot.db")
if not os.path.exists(DB_PATH):
    # Try project root if running from src/ai
    parent_db = "../../flaicheckbot.db"
    if os.path.exists(parent_db):
        DB_PATH = parent_db

print(f"DEBUG: Resolved DB_PATH to: {os.path.abspath(DB_PATH)}")

# Models
LANGUAGE_BASE_MODELS = {
    "de": "fhswf/TrOCR_german_handwritten",
    "en": "microsoft/trocr-base-handwritten",
    "fr": "agomberto/trocr-base-handwritten-fr",
    "es": "qantev/trocr-base-spanish"
}
DEFAULT_BASE_MODEL = "microsoft/trocr-base-handwritten"
ADAPTER_BASE_DIR = "./models/lora"

# Vertex AI
PROJECT_ID = os.getenv("VERTEX_PROJECT_ID", "flaicheckbot-project")
LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
VERTEX_MODEL_NAME = "gemini-2.0-flash"
