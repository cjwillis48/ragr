"""Unit test fixtures."""

import pytest
from unittest.mock import MagicMock

from app.models.rag_model import RagModel


@pytest.fixture
def sample_model():
    """RagModel mock with sensible defaults."""
    model = MagicMock(spec=RagModel)
    model.id = 1
    model.owner_id = "user_123"
    model.slug = "test-model"
    model.name = "Test Model"
    model.system_prompt = "You are a test assistant."
    model.chunk_size = 1000
    model.chunk_overlap = 100
    model.similarity_threshold = 0.3
    model.top_k = 15
    model.embedding_model = "voyage-4-lite"
    model.generation_model = "claude-haiku-4-5"
    model.reranker_enabled = True
    model.rerank_model = "rerank-2.5-lite"
    model.rerank_candidates = 60
    model.rerank_threshold = 0.0
    model.keyword_search_enabled = True
    model.history_turns = 10
    model.hosted_chat = True
    model.budget_limit = 10.0
    model.custom_anthropic_key = None
    model.custom_voyage_key = None
    model.is_active = True
    model.deleted_at = None
    model.allowed_origins = []
    model.sample_questions = []
    return model
