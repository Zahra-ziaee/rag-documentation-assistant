from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"

CHROMA_COLLECTION_NAME = "technical_docs"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

RETRIEVAL_K = 4

# Local embedding model - no OpenAI quota needed
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"