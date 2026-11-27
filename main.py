"""Optional command-line indexing entry point for one document."""

from pathlib import Path

from dotenv import load_dotenv

from src.document_loader import load_document
from src.document_scope import new_document_id
from src.text_splitter import split_documents
from src.vector_store import create_vector_store


def main():
    load_dotenv()

    print("=" * 60)
    print("Technical Documentation Search Assistant")
    print("=" * 60)

    file_path = Path(input("Enter document path: ").strip())
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    documents = load_document(file_path)
    chunks = split_documents(documents)
    document_id = new_document_id()
    create_vector_store(chunks, document_id=document_id)

    print(f"Indexed {len(chunks)} chunks in a separate collection.")
    print(f"Document ID: {document_id}")
    print("The Streamlit app processes its own uploaded documents per session.")


if __name__ == "__main__":
    main()
