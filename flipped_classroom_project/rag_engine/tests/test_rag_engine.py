"""
Tests for rag_engine modules: chat.py, indexer.py, retriever.py.
All FAISS and Groq LLM calls are mocked — no real API key or index needed.
"""

import os
import pickle
import tempfile
import pathlib
from unittest.mock import MagicMock, patch, mock_open

from django.test import TestCase, override_settings


# ──────────────────────────────────────────────────────────────
# rag_engine.indexer — pure utility functions
# ──────────────────────────────────────────────────────────────

class ChunkTextTest(TestCase):
    """Tests for the chunk_text() utility in indexer.py."""

    def setUp(self):
        from rag_engine.indexer import chunk_text
        self.chunk_text = chunk_text

    def test_empty_string_returns_no_chunks(self):
        result = self.chunk_text("", source="test", subject_code="DS")
        self.assertEqual(result, [])

    def test_whitespace_only_returns_no_chunks(self):
        result = self.chunk_text("   \n\t  ", source="test")
        self.assertEqual(result, [])

    def test_short_text_returns_one_chunk(self):
        result = self.chunk_text("Hello world", source="notes", subject_code="PY")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "Hello world")
        self.assertEqual(result[0]["source"], "notes")
        self.assertEqual(result[0]["subject"], "PY")

    def test_chunk_fields(self):
        result = self.chunk_text("A" * 100, source="src", subject_code="CN")
        for chunk in result:
            self.assertIn("text", chunk)
            self.assertIn("source", chunk)
            self.assertIn("subject", chunk)

    def test_long_text_creates_multiple_chunks(self):
        long_text = "X " * 500  # 1000 chars
        result = self.chunk_text(long_text, source="src")
        self.assertGreater(len(result), 1)

    def test_chunk_size_not_exceeded(self):
        from rag_engine.indexer import CHUNK_SIZE
        long_text = "A" * 2000
        result = self.chunk_text(long_text, source="src")
        for chunk in result:
            self.assertLessEqual(len(chunk["text"]), CHUNK_SIZE)

    def test_overlap_between_consecutive_chunks(self):
        from rag_engine.indexer import CHUNK_SIZE, CHUNK_OVERLAP
        # Text longer than 2 chunks
        long_text = "B" * (CHUNK_SIZE * 2)
        result = self.chunk_text(long_text, source="src")
        if len(result) >= 2:
            tail_of_first = result[0]["text"][-CHUNK_OVERLAP:]
            start_of_second = result[1]["text"][:CHUNK_OVERLAP]
            self.assertEqual(tail_of_first, start_of_second)

    def test_subject_code_stored_correctly(self):
        result = self.chunk_text("Some text", source="src", subject_code="AIML")
        self.assertEqual(result[0]["subject"], "AIML")

    def test_default_subject_code_empty(self):
        result = self.chunk_text("Some text", source="src")
        self.assertEqual(result[0]["subject"], "")


class DetectSubjectTest(TestCase):
    """Tests for the _detect_subject() utility in indexer.py."""

    def setUp(self):
        from rag_engine.indexer import _detect_subject
        self.detect = _detect_subject

    def test_detect_ds(self):
        self.assertEqual(self.detect("DS_notes.pdf"), "DS")

    def test_detect_py(self):
        self.assertEqual(self.detect("Python_PY_textbook.pdf"), "PY")

    def test_detect_aiml_before_ai(self):
        # AIML must be detected before a shorter code like AI would be
        self.assertEqual(self.detect("AIML_chapter1.pdf"), "AIML")

    def test_detect_cn(self):
        self.assertEqual(self.detect("Computer_Networks_CN.pdf"), "CN")

    def test_detect_dsc(self):
        self.assertEqual(self.detect("DataScience_DSC.pdf"), "DSC")

    def test_detect_wd(self):
        self.assertEqual(self.detect("WebDev_WD.pdf"), "WD")

    def test_no_code_returns_empty(self):
        self.assertEqual(self.detect("random_textbook.pdf"), "")

    def test_case_insensitive(self):
        self.assertEqual(self.detect("ds_intro.pdf"), "DS")


# ──────────────────────────────────────────────────────────────
# rag_engine.indexer — file loaders (mocked filesystem)
# ──────────────────────────────────────────────────────────────

class LoadKnowledgeFilesTest(TestCase):
    def test_nonexistent_dir_returns_empty(self):
        from rag_engine.indexer import load_knowledge_files
        fake_dir = pathlib.Path("/nonexistent_path_xyz/knowledge")
        result = load_knowledge_files(fake_dir)
        self.assertEqual(result, [])

    def test_loads_txt_file(self):
        from rag_engine.indexer import load_knowledge_files
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            (tmp_path / "DS.txt").write_text("Data structures are important.", encoding="utf-8")
            result = load_knowledge_files(tmp_path)
        self.assertGreater(len(result), 0)
        self.assertEqual(result[0]["subject"], "DS")

    def test_ignores_non_txt_files(self):
        from rag_engine.indexer import load_knowledge_files
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            (tmp_path / "README.md").write_text("# Readme", encoding="utf-8")
            result = load_knowledge_files(tmp_path)
        self.assertEqual(result, [])


class LoadTextbooksTest(TestCase):
    def test_nonexistent_dir_returns_empty(self):
        from rag_engine.indexer import load_textbooks
        fake_dir = pathlib.Path("/nonexistent_path_xyz/textbooks")
        result = load_textbooks(fake_dir)
        self.assertEqual(result, [])

    def test_empty_dir_returns_empty(self):
        from rag_engine.indexer import load_textbooks
        with tempfile.TemporaryDirectory() as tmp:
            result = load_textbooks(pathlib.Path(tmp))
        self.assertEqual(result, [])


class LoadPdfMaterialsTest(TestCase):
    def test_nonexistent_dir_returns_empty(self):
        from rag_engine.indexer import load_pdf_materials
        fake_dir = pathlib.Path("/nonexistent_path_xyz/media")
        result = load_pdf_materials(fake_dir)
        self.assertEqual(result, [])

    def test_loads_txt_file_from_materials(self):
        from rag_engine.indexer import load_pdf_materials
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            mats = tmp_path / "materials"
            mats.mkdir()
            (mats / "notes.txt").write_text("This is a text note.", encoding="utf-8")
            result = load_pdf_materials(tmp_path)
        self.assertGreater(len(result), 0)
        self.assertIn("notes.txt", result[0]["source"])

    def test_loads_md_file_from_materials(self):
        from rag_engine.indexer import load_pdf_materials
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            mats = tmp_path / "materials"
            mats.mkdir()
            (mats / "notes.md").write_text("# Markdown notes content here", encoding="utf-8")
            result = load_pdf_materials(tmp_path)
        self.assertGreater(len(result), 0)


# ──────────────────────────────────────────────────────────────
# rag_engine.chat — build_prompt_messages
# ──────────────────────────────────────────────────────────────

class BuildPromptMessagesTest(TestCase):
    def setUp(self):
        from rag_engine.chat import build_prompt_messages
        self.build = build_prompt_messages

    def test_returns_list_with_system_and_user(self):
        chunks = [{"text": "BFS explores level by level.", "source": "DS notes"}]
        msgs = self.build("What is BFS?", chunks)
        roles = [m["role"] for m in msgs]
        self.assertIn("system", roles)
        self.assertIn("user", roles)

    def test_context_included_in_user_message(self):
        chunks = [{"text": "BFS is Breadth-First Search.", "source": "notes"}]
        msgs = self.build("What is BFS?", chunks)
        user_msg = next(m for m in msgs if m["role"] == "user")
        self.assertIn("BFS is Breadth-First Search.", user_msg["content"])

    def test_no_chunks_uses_no_context_found(self):
        msgs = self.build("What is BFS?", [])
        user_msg = next(m for m in msgs if m["role"] == "user")
        self.assertIn("No specific context found.", user_msg["content"])

    def test_chat_history_included(self):
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]
        msgs = self.build("New question", [], chat_history=history)
        roles = [m["role"] for m in msgs]
        self.assertIn("user", roles)
        self.assertIn("assistant", roles)

    def test_lang_pref_injected_in_user_message(self):
        msgs = self.build("What is BFS?", [], lang_pref="Java")
        user_msg = next(m for m in msgs if m["role"] == "user")
        self.assertIn("Java", user_msg["content"])

    def test_context_truncated_at_max_chars(self):
        from rag_engine.chat import MAX_CONTEXT_CHARS
        big_chunk = {"text": "A" * (MAX_CONTEXT_CHARS + 500), "source": "big"}
        msgs = self.build("question", [big_chunk])
        user_msg = next(m for m in msgs if m["role"] == "user")
        # Context should not exceed max
        self.assertLessEqual(len(user_msg["content"]), len("A" * MAX_CONTEXT_CHARS) + 1000)

    def test_recent_history_only(self):
        """Only last 6 messages from history are included."""
        history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(20)
        ]
        msgs = self.build("Q", [], chat_history=history)
        # system + last 6 history + user = 8 max
        self.assertLessEqual(len(msgs), 8)


# ──────────────────────────────────────────────────────────────
# rag_engine.chat — ask() function
# ──────────────────────────────────────────────────────────────

class AskFunctionTest(TestCase):
    @patch("rag_engine.chat.get_context", return_value=[])
    def test_no_context_returns_not_found(self, _mock):
        from rag_engine.chat import ask
        result = ask("What is BFS?")
        self.assertIn("reply", result)
        self.assertIn("not found", result["reply"].lower())
        self.assertIsNone(result["error"])

    @patch("rag_engine.chat.get_context", return_value=[
        {"text": "BFS explores level by level.", "source": "DS", "subject": "DS", "score": 0.9}
    ])
    @patch("rag_engine.chat._get_groq_client")
    def test_successful_ask(self, mock_client, _mock_ctx):
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "BFS means Breadth-First Search."
        mock_client.return_value.chat.completions.create.return_value = mock_resp
        from rag_engine.chat import ask
        result = ask("What is BFS?")
        self.assertEqual(result["reply"], "BFS means Breadth-First Search.")
        self.assertIsNone(result["error"])
        self.assertIsInstance(result["sources"], list)

    @patch("rag_engine.chat._get_groq_client", side_effect=RuntimeError("API error"))
    @patch("rag_engine.chat.get_context", return_value=[
        {"text": "test text", "source": "src", "subject": "DS", "score": 0.9}
    ])
    def test_groq_error_returns_internal_error(self, _mock_ctx, _mock_client):
        from rag_engine.chat import ask
        result = ask("What is BFS?")
        self.assertEqual(result["error"], "internal_error")

    @patch("rag_engine.chat.get_context", return_value=[])
    def test_greeting_not_found(self, _mock):
        from rag_engine.chat import ask
        for greeting in ["hi", "hello", "hey", "hii", "hola"]:
            result = ask(greeting)
            self.assertIn("not found", result["reply"].lower(), f"Greeting '{greeting}' should not be found")

    @patch("rag_engine.chat.get_context", return_value=[
        {"text": "context text", "source": "src", "subject": "DS", "score": 0.4}
    ])
    @patch("rag_engine.chat._get_groq_client")
    def test_above_threshold_score_triggers_llm(self, mock_client, _mock_ctx):
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "Answer."
        mock_client.return_value.chat.completions.create.return_value = mock_resp
        from rag_engine.chat import ask
        result = ask("What is BFS?")
        # score 0.4 >= 0.38 threshold, so LLM should be called
        mock_client.return_value.chat.completions.create.assert_called()

    @patch("rag_engine.chat.get_context", return_value=[
        {"text": "low score context", "source": "src", "subject": "DS", "score": 0.2}
    ])
    def test_below_threshold_score_returns_not_found(self, _mock_ctx):
        from rag_engine.chat import ask
        result = ask("What is DFS?")
        self.assertIsNone(result["error"])
        self.assertIn("not found", result["reply"].lower())


# ──────────────────────────────────────────────────────────────
# rag_engine.chat — web search helpers
# ──────────────────────────────────────────────────────────────

class WebSearchHelpersTest(TestCase):
    @patch("urllib.request.urlopen")
    def test_fetch_wiki_sources_returns_list(self, mock_urlopen):
        import json
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps([
            "BFS",
            ["Breadth-first search"],
            ["BFS is an algorithm"],
            ["https://en.wikipedia.org/wiki/Breadth-first_search"]
        ]).encode()
        mock_urlopen.return_value.__enter__ = lambda s: mock_response
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)
        from rag_engine.chat import _fetch_wiki_sources
        results = _fetch_wiki_sources("BFS algorithm", limit=1)
        self.assertIsInstance(results, list)

    @patch("urllib.request.urlopen", side_effect=Exception("Network error"))
    def test_fetch_wiki_sources_handles_error(self, _mock):
        from rag_engine.chat import _fetch_wiki_sources
        results = _fetch_wiki_sources("BFS")
        self.assertEqual(results, [])

    @patch("urllib.request.urlopen", side_effect=Exception("Network error"))
    def test_duckduckgo_search_handles_error(self, _mock):
        from rag_engine.chat import _duckduckgo_search
        results = _duckduckgo_search("BFS algorithm")
        self.assertEqual(results, [])

    def test_is_web_search_disabled_by_default(self):
        """Web search should be off unless explicitly enabled."""
        from rag_engine.chat import _is_web_search_enabled
        with override_settings(RAG_ENABLE_WEB_SEARCH=False):
            self.assertFalse(_is_web_search_enabled())

    def test_is_web_search_enabled_when_set(self):
        from rag_engine.chat import _is_web_search_enabled
        with override_settings(RAG_ENABLE_WEB_SEARCH=True):
            self.assertTrue(_is_web_search_enabled())


# ──────────────────────────────────────────────────────────────
# rag_engine.retriever — invalidate_cache and get_context
# ──────────────────────────────────────────────────────────────

class RetrieverTest(TestCase):
    def test_invalidate_cache_resets_globals(self):
        from rag_engine import retriever
        retriever._index = MagicMock()
        retriever._chunks = [{"text": "some chunk"}]
        retriever.invalidate_cache()
        self.assertIsNone(retriever._index)
        self.assertIsNone(retriever._chunks)

    def test_get_context_no_index_returns_empty(self):
        """If index files don't exist, get_context should return []."""
        from rag_engine import retriever
        # Reset to None to force reload
        retriever._index = None
        retriever._chunks = None
        retriever._model = None
        with patch("rag_engine.retriever.INDEX_DIR", pathlib.Path("/nonexistent/path")):
            result = retriever.get_context("What is BFS?")
        self.assertEqual(result, [])

    def test_get_context_with_mock_index(self):
        """get_context with a mocked FAISS index returns chunks."""
        import numpy as np
        from rag_engine import retriever

        # Set up mock index and chunks
        mock_index = MagicMock()
        mock_index.ntotal = 1
        mock_index.search.return_value = (
            np.array([[0.9]], dtype=np.float32),
            np.array([[0]], dtype=np.int64),
        )
        mock_chunks = [{"text": "BFS explores nodes.", "source": "DS", "subject": "DS"}]
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones((1, 384), dtype=np.float32)

        retriever._index = mock_index
        retriever._chunks = mock_chunks
        retriever._model = mock_model

        with patch("faiss.normalize_L2"):
            result = retriever.get_context("BFS", top_k=1)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "BFS explores nodes.")
        self.assertIn("score", result[0])

    def test_get_context_subject_filter(self):
        """Subject filter should exclude non-matching chunks."""
        import numpy as np
        from rag_engine import retriever

        mock_index = MagicMock()
        mock_index.ntotal = 2
        mock_index.search.return_value = (
            np.array([[0.9, 0.8]], dtype=np.float32),
            np.array([[0, 1]], dtype=np.int64),
        )
        mock_chunks = [
            {"text": "DS content", "source": "DS", "subject": "DS"},
            {"text": "PY content", "source": "PY", "subject": "PY"},
        ]
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones((1, 384), dtype=np.float32)

        retriever._index = mock_index
        retriever._chunks = mock_chunks
        retriever._model = mock_model

        with patch("faiss.normalize_L2"):
            result = retriever.get_context("test", top_k=5, subject_filter="DS")

        # Only DS chunk should be returned
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["subject"], "DS")

    def test_get_context_ai_subject_alias(self):
        """Subject code 'AI' should match 'AIML' chunks."""
        import numpy as np
        from rag_engine import retriever

        mock_index = MagicMock()
        mock_index.ntotal = 1
        mock_index.search.return_value = (
            np.array([[0.9]], dtype=np.float32),
            np.array([[0]], dtype=np.int64),
        )
        mock_chunks = [{"text": "AIML content", "source": "AIML", "subject": "AIML"}]
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones((1, 384), dtype=np.float32)

        retriever._index = mock_index
        retriever._chunks = mock_chunks
        retriever._model = mock_model

        with patch("faiss.normalize_L2"):
            result = retriever.get_context("AI question", top_k=5, subject_filter="AI")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["subject"], "AIML")


class CondenseQueryTest(TestCase):
    def test_condense_no_history_returns_original(self):
        from rag_engine.chat import _condense_query
        result = _condense_query("explain in simpler terms", chat_history=None)
        self.assertEqual(result, "explain in simpler terms")

        result_empty = _condense_query("explain in simpler terms", chat_history=[])
        self.assertEqual(result_empty, "explain in simpler terms")

    @patch("rag_engine.chat._get_groq_client")
    def test_condense_with_history_calls_groq(self, mock_get_client):
        from rag_engine.chat import _condense_query
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "Explain BFS in simpler terms"
        mock_client.chat.completions.create.return_value = mock_resp
        mock_get_client.return_value = mock_client

        history = [
            {"role": "user", "content": "What is BFS?"},
            {"role": "assistant", "content": "BFS stands for Breadth First Search..."},
        ]

        result = _condense_query("explain in simpler terms", chat_history=history, client=mock_client)
        self.assertEqual(result, "Explain BFS in simpler terms")
        mock_client.chat.completions.create.assert_called_once()

