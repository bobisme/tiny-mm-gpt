import pytest
import tempfile
import os
from ..mini import Tokenizer, save_tokenizer, load_tokenizer


class TestTokenizer:
    def test_basic_functionality(self):
        """Test basic add, build, encode, decode functionality"""
        tokenizer = Tokenizer()

        # Add some sample text
        tokenizer.add("Hello world!")
        tokenizer.add("This is a test.")
        tokenizer.add("Hello again!")

        # Build the vocabulary
        tokenizer.build()

        # Test encoding
        test_text = "Hello world!"
        tokens = tokenizer.encode(test_text)

        # Test decoding
        decoded = tokenizer.decode(tokens)

        # Verify round-trip
        assert decoded == test_text

        # Verify tokenizer learned some merges
        assert len(tokenizer.merges) > 0

    def test_cannot_add_after_build(self):
        """Test that adding text after build raises error"""
        tokenizer = Tokenizer()
        tokenizer.add("Hello")
        tokenizer.build()

        with pytest.raises(ValueError, match="Vocabulary already built"):
            tokenizer.add("More text")

    def test_cannot_build_twice(self):
        """Test that building twice raises error"""
        tokenizer = Tokenizer()
        tokenizer.add("Hello")
        tokenizer.build()

        with pytest.raises(ValueError, match="Vocabulary already built"):
            tokenizer.build()

    def test_encode_decode_without_build(self):
        """Test that encode/decode without build raises error"""
        tokenizer = Tokenizer()
        tokenizer.add("Hello")

        with pytest.raises(ValueError, match="Vocabulary not built yet"):
            tokenizer.encode("test")

        with pytest.raises(ValueError, match="Vocabulary not built yet"):
            tokenizer.decode([72, 101, 108, 108, 111])

    def test_empty_corpus(self):
        """Test behavior with empty corpus"""
        tokenizer = Tokenizer()
        tokenizer.build()

        # Should still work with basic bytes
        tokens = tokenizer.encode("Hi")
        decoded = tokenizer.decode(tokens)
        assert decoded == "Hi"

    def test_unicode_text(self):
        """Test with unicode text"""
        tokenizer = Tokenizer()
        tokenizer.add("こんにちは")
        tokenizer.add("Hello 世界")
        tokenizer.build()

        test_text = "こんにちは"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        assert decoded == test_text

    def test_save_load_roundtrip(self):
        """Test saving and loading tokenizer preserves functionality"""
        # Create and train original tokenizer
        original = Tokenizer()
        original.add("Hello world!")
        original.add("This is a test.")
        original.add("Hello again!")
        original.build()

        # Test with the original
        test_text = "Hello world!"
        original_tokens = original.encode(test_text)
        original_decoded = original.decode(original_tokens)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name

        try:
            save_tokenizer(original, temp_path)

            # Load from file
            loaded = load_tokenizer(temp_path)

            # Test that loaded tokenizer works the same
            loaded_tokens = loaded.encode(test_text)
            loaded_decoded = loaded.decode(loaded_tokens)

            # Verify same results
            assert loaded_tokens == original_tokens
            assert loaded_decoded == original_decoded
            assert loaded_decoded == test_text

            # Verify internal state matches
            assert len(loaded.merges) == len(original.merges)
            assert loaded.merges == original.merges
            assert loaded._built

        finally:
            os.unlink(temp_path)

    def test_save_unbuilt_tokenizer(self):
        """Test that saving unbuilt tokenizer raises error"""
        tokenizer = Tokenizer()
        tokenizer.add("Hello")

        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name

        try:
            with pytest.raises(
                ValueError, match="Tokenizer must be built before saving"
            ):
                save_tokenizer(tokenizer, temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_invalid_file(self):
        """Test loading invalid file raises appropriate errors"""
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            load_tokenizer("nonexistent.bin")

        # Test with invalid format
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"invalid data")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported file version"):
                load_tokenizer(temp_path)
        finally:
            os.unlink(temp_path)

    def test_empty_tokenizer_save_load(self):
        """Test save/load with empty tokenizer (no merges)"""
        original = Tokenizer()
        original.build()  # Build without adding any text

        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name

        try:
            save_tokenizer(original, temp_path)
            loaded = load_tokenizer(temp_path)

            # Should work with basic byte encoding
            test_text = "Hi"
            original_tokens = original.encode(test_text)
            loaded_tokens = loaded.encode(test_text)

            assert loaded_tokens == original_tokens
            assert loaded.decode(loaded_tokens) == test_text
            assert len(loaded.merges) == 0

        finally:
            os.unlink(temp_path)
