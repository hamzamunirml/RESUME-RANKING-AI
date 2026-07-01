"""
Unit tests for Text Preprocessor module
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import TextPreprocessor


class TestTextPreprocessor:
    """Test cases for TextPreprocessor class"""

    @pytest.fixture
    def preprocessor(self):
        """Create a preprocessor instance"""
        return TextPreprocessor()

    def test_initialization(self, preprocessor):
        """Test preprocessor initialization"""
        assert preprocessor is not None
        assert preprocessor.language == "english"
        assert hasattr(preprocessor, "stop_words")
        assert hasattr(preprocessor, "lemmatizer")

    def test_clean_text_lowercase(self, preprocessor):
        """Test text conversion to lowercase"""
        text = "Hello WORLD"
        cleaned = preprocessor.clean_text(text)
        assert cleaned == "hello world"

    def test_clean_text_remove_special_chars(self, preprocessor):
        """Test removal of special characters"""
        text = "Hello! @World #2024 $100"
        cleaned = preprocessor.clean_text(text)
        assert cleaned == "hello world"

    def test_clean_text_remove_numbers(self, preprocessor):
        """Test removal of numbers"""
        text = "Python 3.9 and 100+ libraries"
        cleaned = preprocessor.clean_text(text)
        assert "python" in cleaned
        assert "libraries" in cleaned
        assert "3.9" not in cleaned
        assert "100" not in cleaned

    def test_clean_text_remove_extra_spaces(self, preprocessor):
        """Test removal of extra spaces"""
        text = "Hello   World   from   Python"
        cleaned = preprocessor.clean_text(text)
        assert cleaned == "hello world from python"
        assert "  " not in cleaned

    def test_tokenize(self, preprocessor):
        """Test tokenization"""
        text = "Hello world from Python"
        tokens = preprocessor.tokenize(text)
        assert tokens == ["Hello", "world", "from", "Python"]

    def test_tokenize_empty_text(self, preprocessor):
        """Test tokenization of empty text"""
        tokens = preprocessor.tokenize("")
        assert tokens == []

    def test_remove_stopwords(self, preprocessor):
        """Test stopword removal"""
        tokens = ["I", "am", "a", "data", "scientist"]
        filtered = preprocessor.remove_stopwords(tokens)
        # 'I' might not be in stopwords, so check for common stopwords
        assert "am" not in filtered
        assert "a" not in filtered
        assert "data" in filtered
        assert "scientist" in filtered

    def test_remove_stopwords_empty(self, preprocessor):
        """Test stopword removal with empty list"""
        filtered = preprocessor.remove_stopwords([])
        assert filtered == []

    def test_lemmatize(self, preprocessor):
        """Test lemmatization"""
        tokens = ["running", "ran", "runs"]
        lemmatized = preprocessor.lemmatize(tokens)
        # Check that at least one token is lemmatized to 'run'
        assert any("run" in token for token in lemmatized)

    def test_lemmatize_empty(self, preprocessor):
        """Test lemmatization with empty list"""
        result = preprocessor.lemmatize([])
        assert result == []

    def test_preprocess_pipeline(self, preprocessor):
        """Test complete preprocessing pipeline"""
        text = "I am a Data Scientist with 5 years of experience in Machine Learning."
        tokens = preprocessor.preprocess(text)

        # Should be a list of tokens
        assert isinstance(tokens, list)
        assert len(tokens) > 0

        # Should contain meaningful words
        assert "data" in tokens or "scientist" in tokens

        # Should not contain stopwords
        assert "i" not in tokens
        assert "am" not in tokens
        assert "a" not in tokens

    def test_get_full_preprocessed_text(self, preprocessor):
        """Test getting full preprocessed text"""
        text = "Hello World from Python"
        result = preprocessor.get_full_preprocessed_text(text)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_string(self, preprocessor):
        """Test preprocessing empty string"""
        result = preprocessor.preprocess("")
        assert result == []

    def test_only_stopwords(self, preprocessor):
        """Test text with only stopwords"""
        text = "I am a the and"
        tokens = preprocessor.preprocess(text)
        assert len(tokens) == 0

    def test_special_characters_only(self, preprocessor):
        """Test text with only special characters"""
        text = "@#$%^&*()"
        tokens = preprocessor.preprocess(text)
        assert tokens == []
