import pytest
from cryptography.fernet import InvalidToken

import app.services.crypto as crypto_module
from app.services.crypto import encrypt, decrypt


@pytest.fixture(autouse=True)
def reset_fernet():
    """Reset the module-level Fernet instance between tests."""
    crypto_module._fernet = None
    yield
    crypto_module._fernet = None


class TestCrypto:
    def test_encrypt_decrypt_roundtrip(self):
        plaintext = "my-secret-api-key"
        ciphertext = encrypt(plaintext)
        assert decrypt(ciphertext) == plaintext

    def test_encrypt_produces_different_ciphertexts(self):
        """Fernet uses random IV, so same plaintext -> different ciphertext."""
        ct1 = encrypt("same-value")
        ct2 = encrypt("same-value")
        assert ct1 != ct2
        # But both decrypt to the same value
        assert decrypt(ct1) == decrypt(ct2) == "same-value"

    def test_decrypt_invalid_token_raises(self):

        with pytest.raises(InvalidToken):
            decrypt("not-valid-ciphertext")

    def test_empty_string_roundtrip(self):
        assert decrypt(encrypt("")) == ""

    def test_unicode_roundtrip(self):
        plaintext = "key with unicode: \u2603\u2764"
        assert decrypt(encrypt(plaintext)) == plaintext
