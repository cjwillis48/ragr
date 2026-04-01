import pytest
from unittest.mock import MagicMock, patch

import app.services.r2 as r2_module
from app.services.r2 import is_configured, _generate_presigned_upload_url, _download_object, _delete_object


@pytest.fixture(autouse=True)
def reset_r2_client():
    r2_module._client = None
    yield
    r2_module._client = None


class TestIsConfigured:
    def test_configured(self):
        with patch.object(r2_module, "settings") as mock_settings:
            mock_settings.r2_account_id = "abc123"
            mock_settings.r2_access_key_id = "key123"
            assert is_configured() is True

    def test_not_configured_empty(self):
        with patch.object(r2_module, "settings") as mock_settings:
            mock_settings.r2_account_id = ""
            mock_settings.r2_access_key_id = ""
            assert is_configured() is False

    def test_partial_config(self):
        with patch.object(r2_module, "settings") as mock_settings:
            mock_settings.r2_account_id = "abc"
            mock_settings.r2_access_key_id = ""
            assert is_configured() is False


class TestPresignedUrl:
    def test_generate_presigned_upload_url(self):
        mock_client = MagicMock()
        mock_client.generate_presigned_url.return_value = "https://r2.example.com/presigned"

        with patch.object(r2_module, "_get_client", return_value=mock_client):
            url = _generate_presigned_upload_url("uploads/test/file.pdf", "application/pdf")

        assert url == "https://r2.example.com/presigned"
        mock_client.generate_presigned_url.assert_called_once_with(
            "put_object",
            Params={
                "Bucket": r2_module.settings.r2_bucket_name,
                "Key": "uploads/test/file.pdf",
                "ContentType": "application/pdf",
            },
            ExpiresIn=r2_module.settings.r2_presigned_expiry,
        )


class TestDownloadObject:
    def test_download_object(self):
        mock_body = MagicMock()
        mock_body.read.return_value = b"file contents"

        mock_client = MagicMock()
        mock_client.get_object.return_value = {"Body": mock_body}

        with patch.object(r2_module, "_get_client", return_value=mock_client):
            data = _download_object("uploads/test/file.pdf")

        assert data == b"file contents"
        mock_body.close.assert_called_once()


class TestDeleteObject:
    def test_delete_object(self):
        mock_client = MagicMock()

        with patch.object(r2_module, "_get_client", return_value=mock_client):
            _delete_object("uploads/test/file.pdf")

        mock_client.delete_object.assert_called_once_with(
            Bucket=r2_module.settings.r2_bucket_name,
            Key="uploads/test/file.pdf",
        )
