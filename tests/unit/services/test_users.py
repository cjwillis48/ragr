from unittest.mock import AsyncMock, MagicMock

from app.services.users import owner_can_use_global_keys


class TestOwnerCanUseGlobalKeys:
    async def test_no_row_returns_false(self):
        session = AsyncMock()
        result = AsyncMock()
        result.scalar_one_or_none = MagicMock(return_value=None)
        session.execute = AsyncMock(return_value=result)

        assert await owner_can_use_global_keys(session, "user_xyz") is False

    async def test_flag_true_returns_true(self):
        session = AsyncMock()
        result = AsyncMock()
        result.scalar_one_or_none = MagicMock(return_value=True)
        session.execute = AsyncMock(return_value=result)

        assert await owner_can_use_global_keys(session, "user_xyz") is True

    async def test_flag_false_returns_false(self):
        session = AsyncMock()
        result = AsyncMock()
        result.scalar_one_or_none = MagicMock(return_value=False)
        session.execute = AsyncMock(return_value=result)

        assert await owner_can_use_global_keys(session, "user_xyz") is False
