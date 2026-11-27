"""Offline unit tests for document isolation; no model downloads are needed."""

import importlib
import sys
import types
import unittest
from unittest.mock import patch

from src.document_scope import collection_name_for, new_document_id


class FakeDocument:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = dict(metadata or {})


class FakeChroma:
    """A minimal in-memory stand-in for the Chroma collection API."""

    registry = {}

    def __init__(self, *, collection_name, embedding_function, persist_directory):
        self.collection_name = collection_name
        self.registry.setdefault(collection_name, [])

    @classmethod
    def from_documents(
        cls, *, documents, embedding, collection_name, persist_directory
    ):
        store = cls(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=persist_directory,
        )
        cls.registry[collection_name].extend(documents)
        return store

    def as_retriever(self, *, search_kwargs):
        return self

    def invoke(self, question):
        return list(self.registry[self.collection_name])


class FakeEmbeddings:
    def __init__(self, **kwargs):
        pass

    @staticmethod
    def _vector(text):
        text = text.lower()
        if any(word in text for word in ("python", "function", "defined")):
            return [1.0, 0.0, 0.0]
        if any(word in text for word in ("sql", "join", "group by")):
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def embed_query(self, text):
        return self._vector(text)

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]


def fake_imports():
    chroma_module = types.ModuleType("langchain_chroma")
    chroma_module.Chroma = FakeChroma
    core_module = types.ModuleType("langchain_core")
    documents_module = types.ModuleType("langchain_core.documents")
    documents_module.Document = FakeDocument
    embeddings_module = types.ModuleType("langchain_huggingface")
    embeddings_module.HuggingFaceEmbeddings = FakeEmbeddings
    return {
        "langchain_chroma": chroma_module,
        "langchain_core": core_module,
        "langchain_core.documents": documents_module,
        "langchain_huggingface": embeddings_module,
    }


class TestDocumentScope(unittest.TestCase):
    def test_document_ids_are_unique_and_well_formed(self):
        first, second = new_document_id(), new_document_id()
        self.assertNotEqual(first, second)
        self.assertEqual(len(first), 32)
        self.assertEqual(collection_name_for(first), f"doc_{first}")

    def test_invalid_document_ids_are_rejected(self):
        for invalid in [None, "", "technical_docs", "../other", "A" * 32]:
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                collection_name_for(invalid)

    def test_two_documents_cannot_cross_retrieve(self):
        FakeChroma.registry.clear()
        with patch.dict(sys.modules, fake_imports()):
            # Reload only the modules being tested against lightweight stand-ins.
            sys.modules.pop("src.vector_store", None)
            sys.modules.pop("src.rag_chain", None)
            vs = importlib.import_module("src.vector_store")
            chain = importlib.import_module("src.rag_chain")
            try:
                a, b = new_document_id(), new_document_id()
                vs.create_vector_store(
                    [FakeDocument("Python functions", {"source": "python.txt", "chunk_id": 1})],
                    document_id=a,
                )
                vs.create_vector_store(
                    [FakeDocument("SQL joins", {"source": "sql.txt", "chunk_id": 1})],
                    document_id=b,
                )

                result_a = chain.answer_question("explain functions", document_id=a)
                result_b = chain.answer_question("explain joins", document_id=b)
                self.assertIn("Python functions", result_a["answer"])
                self.assertNotIn("SQL joins", result_a["answer"])
                self.assertIn("SQL joins", result_b["answer"])
                self.assertNotIn("Python functions", result_b["answer"])
                self.assertEqual(result_a["sources"][0]["source"], "python.txt")
                self.assertEqual(result_b["sources"][0]["source"], "sql.txt")
            finally:
                sys.modules.pop("src.vector_store", None)
                sys.modules.pop("src.rag_chain", None)

    def test_unrelated_question_returns_no_sources(self):
        """An earthquake question must not return SQL excerpts."""
        FakeChroma.registry.clear()
        with patch.dict(sys.modules, fake_imports()):
            sys.modules.pop("src.vector_store", None)
            sys.modules.pop("src.rag_chain", None)
            vs = importlib.import_module("src.vector_store")
            chain = importlib.import_module("src.rag_chain")
            try:
                document_id = new_document_id()
                vs.create_vector_store(
                    [FakeDocument(
                        "SQL JOIN combines tables; GROUP BY aggregates rows.",
                        {"source": "sql_demo.txt", "chunk_id": 1},
                    )], document_id=document_id,
                )
                result = chain.answer_question(
                    "What causes earthquakes?", document_id=document_id
                )
                self.assertEqual(result["sources"], [])
                self.assertIn("could not find sufficiently relevant", result["answer"])
            finally:
                sys.modules.pop("src.vector_store", None)
                sys.modules.pop("src.rag_chain", None)

    def test_related_question_keeps_source_and_similarity(self):
        FakeChroma.registry.clear()
        with patch.dict(sys.modules, fake_imports()):
            sys.modules.pop("src.vector_store", None)
            sys.modules.pop("src.rag_chain", None)
            vs = importlib.import_module("src.vector_store")
            chain = importlib.import_module("src.rag_chain")
            try:
                document_id = new_document_id()
                vs.create_vector_store(
                    [FakeDocument(
                        "A SQL JOIN combines rows from tables.",
                        {"source": "sql_demo.txt", "chunk_id": 1},
                    )], document_id=document_id,
                )
                result = chain.answer_question(
                    "What does JOIN do?", document_id=document_id
                )
                self.assertEqual(len(result["sources"]), 1)
                self.assertEqual(result["sources"][0]["source"], "sql_demo.txt")
                self.assertAlmostEqual(result["sources"][0]["relevance_score"], 1.0)
            finally:
                sys.modules.pop("src.vector_store", None)
                sys.modules.pop("src.rag_chain", None)

    def test_cosine_scores_independent_of_vector_size(self):
        with patch.dict(sys.modules, fake_imports()):
            sys.modules.pop("src.vector_store", None)
            vs = importlib.import_module("src.vector_store")
            try:
                self.assertAlmostEqual(
                    vs.cosine_similarity([2, 0], [10, 0]), 1.0
                )
                self.assertAlmostEqual(
                    vs.cosine_similarity([1, 0], [0, 1]), 0.0
                )
                self.assertAlmostEqual(
                    vs.cosine_similarity([1, 0], [-1, 0]), -1.0
                )
            finally:
                sys.modules.pop("src.vector_store", None)

    def test_empty_question_or_chunks_are_rejected(self):
        with patch.dict(sys.modules, fake_imports()):
            sys.modules.pop("src.vector_store", None)
            vs = importlib.import_module("src.vector_store")
            try:
                document_id = new_document_id()
                with self.assertRaises(ValueError):
                    vs.create_vector_store([], document_id=document_id)
                with self.assertRaises(ValueError):
                    vs.retrieve_relevant_chunks("   ", document_id=document_id)
            finally:
                sys.modules.pop("src.vector_store", None)


if __name__ == "__main__":
    unittest.main()
