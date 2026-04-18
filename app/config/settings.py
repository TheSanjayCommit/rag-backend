import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    COLLEGE_API_KEY: str = os.getenv("COLLEGE_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    
    # RAG Settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    FAISS_INDEX_PATH: str = os.path.join(os.path.dirname(__file__), "..", "db", "faiss_index")
    DATA_PATH: str = os.path.join(os.path.dirname(__file__), "..", "data")
    
    # App Settings
    PORT: int = 8000
    DEBUG: bool = False   # Set DEBUG=True in .env for local development only

settings = Settings()
