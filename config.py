import os
from dotenv import load_dotenv

load_dotenv()

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE')

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')

# Qdrant Configuration
QDRANT_URL = os.getenv('QDRANT_URL', ':memory:')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY', '')

# Model Configuration
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2:7b"

# RAG backend: "langchain" (исходный пайплайн) или "llamaindex" (llamaindex_pipeline.py)
RAG_BACKEND = os.getenv('RAG_BACKEND', 'langchain')

# Реранкинг: пусто = выключен; иначе имя cross-encoder модели (напр. BAAI/bge-reranker-v2-m3)
RERANKER_MODEL = os.getenv('RERANKER_MODEL', '')
RERANK_CANDIDATES = int(os.getenv('RERANK_CANDIDATES', '20'))  # сколько достаём из Qdrant до реранкинга
TOP_K = int(os.getenv('TOP_K', '3'))  # сколько чанков уходит в промпт

# Text Processing
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_TOKENS = 4000

# Chat Configuration
CHAT_SYSTEM_PROMPT = "ты общаешься в телеграм, отвечай кратко в 1-2 предложения, будь естественным, как будто пишешь другу"
MAX_HISTORY_LENGTH = 50  # Максимальное количество сообщений в истории