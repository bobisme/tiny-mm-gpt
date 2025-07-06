import pytest
from ..mini import Tokenizer


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
