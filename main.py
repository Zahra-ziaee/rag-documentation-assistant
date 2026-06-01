from pathlib import Path

from dotenv import load_dotenv

from src.document_loader import load_document
from src.text_splitter import split_documents
from src.vector_store import create_vector_store


def main():
    load_dotenv()

    print("=" * 60)
    print("RAG Technical Documentation Assistant")
    print("=" * 60)

    file_path_input = input("Enter document path: ").strip()
    file_path = Path(file_path_input)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    print("\nLoading document...")
    documents = load_document(file_path)
    print(f"Loaded documents/pages: {len(documents)}")

    print("\nSplitting document into chunks...")
    chunks = split_documents(documents)
    print(f"Created chunks: {len(chunks)}")

    print("\nCreating vector store...")
    create_vector_store(chunks)
    print("Vector store created successfully.")


if __name__ == "__main__":
    main()