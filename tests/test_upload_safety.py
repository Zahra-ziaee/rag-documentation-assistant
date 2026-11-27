"""Upload path and size tests without importing LangChain dependencies."""

import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class FakeUpload:
    def __init__(self, name, data):
        self.name = name
        self.data = data

    def getbuffer(self):
        return memoryview(self.data)


class TestUploadSafety(unittest.TestCase):
    def setUp(self):
        community = types.ModuleType("langchain_community")
        loaders = types.ModuleType("langchain_community.document_loaders")
        loaders.PyPDFLoader = type("PyPDFLoader", (), {})
        loaders.TextLoader = type("TextLoader", (), {})
        core = types.ModuleType("langchain_core")
        documents = types.ModuleType("langchain_core.documents")
        documents.Document = type("Document", (), {})
        self.mocked_deps = patch.dict(
            sys.modules,
            {
                "langchain_community": community,
                "langchain_community.document_loaders": loaders,
                "langchain_core": core,
                "langchain_core.documents": documents,
            },
        )
        self.mocked_deps.start()
        sys.modules.pop("src.document_loader", None)
        self.loader = importlib.import_module("src.document_loader")
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path_patcher = patch.object(
            self.loader, "DOCUMENTS_DIR", Path(self.temp_dir.name)
        )
        self.path_patcher.start()
        self.addCleanup(self.path_patcher.stop)
        self.addCleanup(self.cleanup_import)

    def cleanup_import(self):
        sys.modules.pop("src.document_loader", None)
        self.mocked_deps.stop()

    def test_same_named_uploads_do_not_overwrite(self):
        first = self.loader.save_uploaded_file(FakeUpload("notes.txt", b"first"))
        second = self.loader.save_uploaded_file(FakeUpload("notes.txt", b"second"))
        self.assertNotEqual(first, second)
        self.assertEqual(first.read_bytes(), b"first")
        self.assertEqual(second.read_bytes(), b"second")

    def test_path_traversal_does_not_leave_documents_directory(self):
        path = self.loader.save_uploaded_file(
            FakeUpload("../../private/notes.txt", b"safe")
        )
        self.assertEqual(path.parent, Path(self.temp_dir.name))
        self.assertEqual(path.suffix, ".txt")

    def test_reject_unsupported_empty_and_oversized(self):
        with self.assertRaises(ValueError):
            self.loader.save_uploaded_file(FakeUpload("script.py", b"print(1)"))
        with self.assertRaises(ValueError):
            self.loader.save_uploaded_file(FakeUpload("empty.txt", b""))
        with patch.object(self.loader, "MAX_UPLOAD_BYTES", 3):
            with self.assertRaises(ValueError):
                self.loader.save_uploaded_file(FakeUpload("large.md", b"1234"))


if __name__ == "__main__":
    unittest.main()
