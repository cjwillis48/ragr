import pytest
from unittest.mock import MagicMock

from app.services.generation import _parse_meta, _build_prompt


class TestParseMeta:
    def test_answered(self):
        raw = 'Hello there!\n<meta status="answered" />'
        answer, status = _parse_meta(raw)
        assert answer == "Hello there!"
        assert status == "answered"

    def test_unanswered(self):
        raw = 'Sorry, I cannot help.\n<meta status="unanswered" />'
        answer, status = _parse_meta(raw)
        assert answer == "Sorry, I cannot help."
        assert status == "unanswered"

    def test_off_topic(self):
        raw = 'That is outside my scope.\n<meta status="off_topic" />'
        answer, status = _parse_meta(raw)
        assert answer == "That is outside my scope."
        assert status == "off_topic"

    def test_missing_tag_defaults_to_answered(self):
        raw = "Just a plain answer with no meta tag."
        answer, status = _parse_meta(raw)
        assert answer == raw
        assert status == "answered"

    def test_whitespace_around_tag(self):
        raw = 'Answer text.  \n  <meta status="answered" />  '
        answer, status = _parse_meta(raw)
        assert answer == "Answer text."
        assert status == "answered"

    def test_tag_only(self):
        raw = '<meta status="answered" />'
        answer, status = _parse_meta(raw)
        assert answer == ""
        assert status == "answered"


class TestBuildPrompt:
    def _make_chunk(self, content: str, source_url: str = ""):
        chunk = MagicMock()
        chunk.content = content
        chunk.source_url = source_url
        return chunk

    def test_basic_structure(self, sample_model):
        chunks = [self._make_chunk("chunk content")]
        system, messages = _build_prompt(sample_model, "What is X?", chunks)

        # System should be a list with one dict containing text
        assert isinstance(system, list)
        assert len(system) == 1
        assert "You are a test assistant." in system[0]["text"]
        assert "cache_control" in system[0]

        # Messages should end with user message containing knowledge tags
        assert messages[-1]["role"] == "user"
        assert "<knowledge>" in messages[-1]["content"]
        assert "chunk content" in messages[-1]["content"]
        assert "What is X?" in messages[-1]["content"]

    def test_with_http_source_url(self, sample_model):
        chunks = [self._make_chunk("content", "https://example.com/page")]
        _, messages = _build_prompt(sample_model, "Q?", chunks)
        assert "[ref: https://example.com/page]" in messages[-1]["content"]

    def test_without_http_source_url(self, sample_model):
        chunks = [self._make_chunk("content", "file:doc.pdf")]
        _, messages = _build_prompt(sample_model, "Q?", chunks)
        assert "[ref:" not in messages[-1]["content"]

    def test_empty_source_url(self, sample_model):
        chunks = [self._make_chunk("content", "")]
        _, messages = _build_prompt(sample_model, "Q?", chunks)
        assert "[ref:" not in messages[-1]["content"]

    def test_with_history(self, sample_model):
        chunks = [self._make_chunk("ctx")]
        history = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]
        _, messages = _build_prompt(sample_model, "follow up?", chunks, history=history)

        assert len(messages) == 3  # 2 history + 1 current
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "previous question"
        assert messages[1]["role"] == "assistant"
        assert messages[-1]["role"] == "user"
        assert "follow up?" in messages[-1]["content"]

    def test_no_system_prompt_uses_default(self, sample_model):
        sample_model.system_prompt = None
        chunks = [self._make_chunk("ctx")]
        system, _ = _build_prompt(sample_model, "Q?", chunks)
        assert "You are a helpful assistant." in system[0]["text"]

    def test_multiple_chunks_separated(self, sample_model):
        chunks = [self._make_chunk("chunk1"), self._make_chunk("chunk2")]
        _, messages = _build_prompt(sample_model, "Q?", chunks)
        content = messages[-1]["content"]
        assert "chunk1" in content
        assert "chunk2" in content
        assert "---" in content  # chunks separated by dividers
