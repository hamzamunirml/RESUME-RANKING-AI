"""
Text Preprocessing Module - Cleans and prepares text for analysis
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from typing import List, Optional

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")


class TextPreprocessor:
    """
    A class to preprocess and clean text data
    """

    def __init__(self, language: str = "english"):
        """
        Initialize the preprocessor

        Args:
            language: Language for stopwords (default: 'english')
        """
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

        # Additional custom stopwords
        self.custom_stopwords = {
            "resume",
            "curriculum",
            "vitae",
            "cv",
            "page",
            "experience",
            "education",
            "skills",
            "summary",
            "profile",
            "objective",
            "professional",
            "work",
            "job",
            "position",
            "company",
        }
        self.stop_words.update(self.custom_stopwords)

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters, numbers, and extra spaces

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove special characters and digits (keep letters and spaces)
        text = re.sub(r"[^a-zA-Z\s]", " ", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        if not text:
            return []

        # Use NLTK's word tokenizer
        tokens = nltk.word_tokenize(text)
        return tokens

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list

        Args:
            tokens: List of tokens

        Returns:
            Filtered tokens
        """
        if not tokens:
            return []

        return [token for token in tokens if token not in self.stop_words]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens to base form

        Args:
            tokens: List of tokens

        Returns:
            Lemmatized tokens
        """
        if not tokens:
            return []

        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def stem(self, tokens: List[str]) -> List[str]:
        """
        Stem tokens to root form

        Args:
            tokens: List of tokens

        Returns:
            Stemmed tokens
        """
        if not tokens:
            return []

        return [self.stemmer.stem(token) for token in tokens]

    def preprocess(self, text: str, use_lemma: bool = True) -> List[str]:
        """
        Complete preprocessing pipeline

        Args:
            text: Input text
            use_lemma: If True, use lemmatization; else use stemming

        Returns:
            Preprocessed tokens
        """
        # Clean text
        cleaned = self.clean_text(text)

        # Tokenize
        tokens = self.tokenize(cleaned)

        # Remove stopwords
        filtered = self.remove_stopwords(tokens)

        # Lemmatize or stem
        if use_lemma:
            return self.lemmatize(filtered)
        else:
            return self.stem(filtered)

    def get_full_preprocessed_text(self, text: str) -> str:
        """
        Get full preprocessed text as a single string

        Args:
            text: Input text

        Returns:
            Preprocessed text as string
        """
        tokens = self.preprocess(text)
        return " ".join(tokens)


# Test function
def test_preprocessor():
    """Test the preprocessor with sample text"""
    preprocessor = TextPreprocessor()

    sample_text = """
    I am a Data Scientist with 5 years of experience in Machine Learning.
    I have worked on various projects including NLP and Computer Vision.
    """

    print("Original Text:")
    print(sample_text)
    print("\n" + "=" * 50 + "\n")

    cleaned = preprocessor.clean_text(sample_text)
    print("Cleaned Text:")
    print(cleaned)
    print("\n" + "=" * 50 + "\n")

    tokens = preprocessor.preprocess(sample_text)
    print("Preprocessed Tokens:")
    print(tokens)
    print("\n" + "=" * 50 + "\n")

    full_text = preprocessor.get_full_preprocessed_text(sample_text)
    print("Full Preprocessed Text:")
    print(full_text)


if __name__ == "__main__":
    test_preprocessor()
