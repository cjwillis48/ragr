import pytest
from pydantic import ValidationError

from app.schemas.models import (
    RagModelCreate,
    RagModelUpdate,
    ChatTheme,
    SUPPORTED_EMBEDDING_MODELS,
    SUPPORTED_GENERATION_MODELS,
)


class TestRagModelCreate:
    def test_valid_minimal(self):
        m = RagModelCreate(name="Test", slug="test-model")
        assert m.name == "Test"
        assert m.slug == "test-model"
        assert m.description == ""

    def test_valid_with_all_fields(self):
        m = RagModelCreate(
            name="My Bot",
            slug="my-bot",
            description="A test bot",
            system_prompt="Be helpful",
            embedding_model="voyage-4-lite",
            generation_model="claude-haiku-4-5",
            chunk_size=500,
            chunk_overlap=50,
            similarity_threshold=0.5,
            top_k=10,
        )
        assert m.embedding_model == "voyage-4-lite"

    def test_invalid_slug_uppercase(self):
        with pytest.raises(ValidationError, match="slug"):
            RagModelCreate(name="Test", slug="TestModel")

    def test_invalid_slug_special_chars(self):
        with pytest.raises(ValidationError, match="slug"):
            RagModelCreate(name="Test", slug="test_model!")

    def test_invalid_slug_starts_with_dash(self):
        with pytest.raises(ValidationError, match="slug"):
            RagModelCreate(name="Test", slug="-test")

    def test_valid_slug_with_numbers(self):
        m = RagModelCreate(name="Test", slug="model-123")
        assert m.slug == "model-123"

    def test_valid_slug_starts_with_number(self):
        m = RagModelCreate(name="Test", slug="123-model")
        assert m.slug == "123-model"

    def test_unsupported_embedding_model(self):
        with pytest.raises(ValidationError, match="Unsupported model"):
            RagModelCreate(name="Test", slug="test", embedding_model="bad-model")

    def test_unsupported_generation_model(self):
        with pytest.raises(ValidationError, match="Unsupported model"):
            RagModelCreate(name="Test", slug="test", generation_model="gpt-4")

    def test_all_supported_embedding_models(self):
        for model in SUPPORTED_EMBEDDING_MODELS:
            m = RagModelCreate(name="T", slug="t", embedding_model=model)
            assert m.embedding_model == model

    def test_all_supported_generation_models(self):
        for model in SUPPORTED_GENERATION_MODELS:
            m = RagModelCreate(name="T", slug="t", generation_model=model)
            assert m.generation_model == model

    def test_invalid_allowed_origins_with_path(self):
        with pytest.raises(ValidationError, match="Invalid origin"):
            RagModelCreate(name="T", slug="t", allowed_origins=["https://example.com/path"])

    def test_invalid_allowed_origins_trailing_slash(self):
        with pytest.raises(ValidationError, match="Invalid origin"):
            RagModelCreate(name="T", slug="t", allowed_origins=["https://example.com/"])

    def test_valid_allowed_origins(self):
        m = RagModelCreate(
            name="T",
            slug="t",
            allowed_origins=["https://example.com", "http://localhost:3000"],
        )
        assert len(m.allowed_origins) == 2

    def test_chunk_size_bounds(self):
        with pytest.raises(ValidationError):
            RagModelCreate(name="T", slug="t", chunk_size=50)  # min 100
        with pytest.raises(ValidationError):
            RagModelCreate(name="T", slug="t", chunk_size=20000)  # max 10000

    def test_name_required(self):
        with pytest.raises(ValidationError):
            RagModelCreate(slug="test")

    def test_slug_required(self):
        with pytest.raises(ValidationError):
            RagModelCreate(name="Test")


class TestRagModelUpdate:
    def test_partial_update_name_only(self):
        m = RagModelUpdate(name="New Name")
        assert m.name == "New Name"
        assert m.description is None

    def test_all_none_is_valid(self):
        m = RagModelUpdate()
        assert m.name is None

    def test_is_active_field(self):
        m = RagModelUpdate(is_active=False)
        assert m.is_active is False


class TestChatTheme:
    def test_all_none(self):
        t = ChatTheme()
        assert t.label is None

    def test_partial_fields(self):
        t = ChatTheme(primary_color="#ff0000", border_radius=12)
        assert t.primary_color == "#ff0000"
        assert t.border_radius == 12
        assert t.bg_color is None
