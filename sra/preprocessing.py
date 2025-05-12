import re
from collections import Counter
from pathlib import Path

import nltk
import pdfplumber
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure nltk data path and lazy download
nltk.data.path.append("./nltk_data")


class ResumeProcessor:
    """A utility class for processing resume files in .txt or .pdf format.

    This class provides methods to:
    - Load resume content from text or PDF files
    - Preprocess and clean the text (lowercasing, punctuation removal, tokenization,
      stopword removal, and lemmatization)
    - Extract the most frequent keywords from the processed text

    Attributes:
        file_path (str | Path): Absolute or relative path to the resume file.
    """

    def __init__(self, file_path, top_n=10) -> None:
        """Initialize the ResumeProcessor with necessary NLTK resources."""
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.file_path = file_path
        self.top_n = top_n

        self.text = self.load_resume(self.file_path)
        self.tokens = self.clean_text(self.text)
        self.keywords = self.extract_keywords(self.tokens, top_n=self.top_n)

    def load_resume(self, file_path: str | Path) -> str:
        """Load and extract plain text from a resume file (.txt or .pdf).

        Returns:
            str: Extracted plain text content.

        Raises:
            ValueError: If the file format is not supported (only .txt and .pdf are allowed).
        """

        file_path = (
            self.file_path if isinstance(self.file_path, Path) else str(self.file_path)
        )

        ext = Path(file_path).suffix.lower()

        if ext == ".txt":
            with Path(file_path).open(encoding="utf-8") as f:
                return f.read()

        elif ext == ".pdf":
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text

        else:
            msg = "Unsupported file format. Please use .txt or .pdf"
            raise ValueError(msg)

    def clean_text(self, text: str) -> list[str]:
        """Clean and normalize resume text for NLP processing.

        Processing steps:
        - Convert to lowercase
        - Remove punctuation
        - Tokenize into words
        - Remove stopwords
        - Lemmatize each word

        Args:
            text (str): Raw resume text as a string.

        Returns:
            list[str]: List of cleaned and lemmatized word tokens.
        """
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word.isalpha()]
        tokens = [word for word in tokens if word not in self.stop_words]
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def extract_keywords(
        self, tokens: list[str], top_n: int = 10
    ) -> list[tuple[str, int]]:
        """Identify and return the top N most frequent keywords from tokenized text.

        Args:
            tokens (list[str]): List of cleaned, lemmatized tokens.
            top_n (int, optional): Number of top keywords to return. Defaults to 10.

        Returns:
            list[tuple[str, int]]: A list of (keyword, frequency) tuples.
        """
        counter = Counter(tokens)
        return counter.most_common(top_n)
