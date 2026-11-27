"""Load document text and safely persist uploaded files."""

from pathlib import Path
from typing import List
from uuid import uuid4

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from src.config import DOCUMENTS_DIR

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def _safe_display_name(raw_name: str) -> str:
    """Retain the visible base filename, never a directory from the client."""
    if not isinstance(raw_name, str) or not raw_name or "\x00" in raw_name:
        raise ValueError("Invalid uploaded filename.")
    name = raw_name.replace("\\", "/").rsplit("/", 1)[-1]
    if name in {"", ".", ".."}:
        raise ValueError("Invalid uploaded filename.")
    return name


def save_uploaded_file(uploaded_file) -> Path:
    """Validate size/type and store using a collision-resistant server filename."""
    display_name = _safe_display_name(uploaded_file.name)
    suffix = Path(display_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError("Only PDF, TXT, and Markdown files are supported.")

    contents = uploaded_file.getbuffer()
    if not 0 < len(contents) <= MAX_UPLOAD_BYTES:
        raise ValueError("Uploaded file must be between 1 byte and 20 MB.")

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    file_path = DOCUMENTS_DIR / f"{uuid4().hex}{suffix}"
    with file_path.open("wb") as file:
        file.write(contents)
    return file_path


def load_document(file_path: Path, *, source_name: str | None = None) -> List[Document]:
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {suffix}. Supported types: PDF, TXT, MD."
        )

    if suffix == ".pdf":
        documents = PyPDFLoader(str(file_path)).load()
    else:
        documents = TextLoader(str(file_path), encoding="utf-8").load()

    display_name = (
        _safe_display_name(source_name)
        if source_name is not None
        else file_path.name
    )
    for document in documents:
        document.metadata["source"] = display_name
    return documents
