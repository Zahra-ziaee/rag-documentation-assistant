"""Create and validate document identifiers used for isolated Chroma collections."""

import re
from uuid import uuid4

_DOCUMENT_ID_PATTERN = re.compile(r"[0-9a-f]{32}\Z")


def new_document_id() -> str:
    """Return an unpredictable identifier for one processed document."""
    return uuid4().hex


def collection_name_for(document_id: str) -> str:
    """Reject unsafe or missing IDs instead of falling back to a shared collection."""
    if not isinstance(document_id, str) or not _DOCUMENT_ID_PATTERN.fullmatch(
        document_id
    ):
        raise ValueError("document_id must be a 32-character lowercase hex ID.")
    return f"doc_{document_id}"
