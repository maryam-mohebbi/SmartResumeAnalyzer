from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import nltk
import pdfplumber
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sra.utility import get_logger

# Ensure nltk data path and lazy download
nltk.data.path.append("./nltk_data")
nlp = spacy.load("en_core_web_sm")

SKILL_SET = {
    "python",
    "java",
    "c++",
    "sql",
    "machine learning",
    "data analysis",
    "deep learning",
    "nlp",
    "pandas",
    "numpy",
    "git",
    "tensorflow",
    "pytorch",
    "aws",
    "docker",
    "linux",
    "excel",
    "power bi",
    "flask",
}


class ResumeProcessor:
    """A utility class for processing resume files in .txt or .pdf format.

    This class provides methods to:
    - Load resume content from text or PDF files
    - Preprocess and clean the text (lowercasing, punctuation removal, tokenization,
      stopword removal, and lemmatization)
    - Extract the most frequent keywords from the processed text

    Attributes:
        file_path (str | Path): Absolute or relative path to the resume file.
        top_n (int | None): Number of top keywords to extract. Defaults to 10.
        text (str): Raw text content of the resume.
        tokens (list[str]): List of cleaned and lemmatized tokens.
        keywords (list[tuple[str, int]]): List of tuples containing keywords and their
            frequencies.
    """

    def __init__(self, file_path: str | Path, top_n: int | None = 10) -> None:
        """Initialize the ResumeProcessor with necessary NLTK resources.

        Args:
            file_path (str | Path): Path to the resume file.
            top_n (int | None): Number of top keywords to extract. Defaults to 10.
        """
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

        if file_path is None:
            msg = "File path cannot be None, please provide a valid path as string or Path."
            raise ValueError(
                msg,
            )

        self.file_path = file_path
        self.top_n = top_n

        self.text: str = self.load_resume()
        self.tokens: list[str] = self.clean_text()
        self.keywords = self.extract_keywords()

        self.spacy_doc = nlp(self.text)

    def load_resume(self) -> str:
        """Load and extract plain text from a resume file (.txt or .pdf).

        Returns:
            str: Extracted plain text content.

        Raises:
            ValueError: If the file format is not supported (only .txt and .pdf are allowed).
        """
        file_path = self.file_path if isinstance(self.file_path, Path) else str(self.file_path)

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

    def clean_text(self) -> list[str]:
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
        text = self.text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word.isalpha()]
        tokens = [word for word in tokens if word not in self.stop_words]
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def extract_keywords(
        self,
    ) -> list[tuple[str, int]]:
        """Identify and return the top N most frequent keywords from tokenized text.

        Args:
            tokens (list[str]): List of cleaned, lemmatized tokens.

        Returns:
            list[tuple[str, int]]: A list of (keyword, frequency) tuples.
        """
        counter = Counter(self.tokens)
        return counter.most_common(self.top_n)

    def keywords_preview(self) -> str:
        """Return a preview of the top N keywords and their frequencies.

        Returns:
            str: A formatted string of the top N keywords and their frequencies.
        """
        return "\n".join(f"{word}: {freq}" for word, freq in self.keywords)

    def get_section_checklist(self) -> dict[str, str]:
        """Check for the presence of key resume sections such as Email, Name, Skills, Education, Work Experience, and Projects.

        Returns:
            dict[str, str]: A dictionary indicating whether each section is found or likely present in the resume.
        """
        checklist = {}

        text_lower = self.text.lower()

        # Email
        if re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", self.text):
            checklist["Email"] = "Found"
        else:
            checklist["Email"] = "Not Found"

        # Name - only check if there's a line starting with "name"
        checklist["Name"] = "Likely Present" if re.search(r"\bname\s*:", text_lower) else "Unknown"

        # Skills - check for section header or at least 3 matched skills
        skills = self.match_skills()
        min_skills = 3
        if "skills" in text_lower or len(skills) >= min_skills:
            checklist["Skills"] = "Found"
        else:
            checklist["Skills"] = "Not Found"

        # Education - look for common phrases
        if re.search(r"\b(bachelor|master|ph\.?d|education|university|college|b\.sc|m\.sc)\b", text_lower):
            checklist["Education"] = "Found"
        else:
            checklist["Education"] = "Not Found"

        # Experience - look for work, job, or company-related keywords
        if re.search(r"\b(experience|work|intern|employment|company)\b", text_lower):
            checklist["Work Experience"] = "Found"
        else:
            checklist["Work Experience"] = "Not Found"

        # Projects - optional, but common in tech CVs
        if "project" in text_lower:
            checklist["Projects"] = "Found"
        else:
            checklist["Projects"] = "Not Found"

        return checklist

    def match_skills(self) -> list[str]:
        """Match known skills against the resume text.

        Returns:
            list[str]: Detected skills.
        """
        text = " ".join(self.tokens)
        found = [skill for skill in SKILL_SET if skill in text]
        return sorted(found)

    def extract_pos_tags(self) -> list[tuple[str, str]]:
        """Return a list of tokens with their POS tags."""
        return [(token.text, token.pos_) for token in self.spacy_doc]

    def extract_named_entities(self) -> list[tuple[str, str]]:
        """Return named entities with their labels."""
        return [(ent.text, ent.label_) for ent in self.spacy_doc.ents]


def folder_resume_processor(
    folder_path: str | Path,
    top_n: int | None = None,
) -> list[tuple[str, list[tuple[str, int]]]]:
    """Process all resumes in a folder and extract top keywords.

    Args:
        folder_path (str | Path): Path to the folder containing resume files.
        top_n (int | None): Number of top keywords to extract from each resume. Defaults to None.

    Returns:
        list[tuple[str, list[tuple[str, int]]]]: A list of tuples where each tuple contains
            the file name and a list of (keyword, frequency) pairs.
    """
    logger = get_logger("Folder Resume Processor")

    folder = Path(folder_path)
    if not folder.exists():
        msg = "Folder does not exist: %s", folder_path
        raise ValueError(msg)

    resume_files = list(folder.glob("*.pdf")) + list(folder.glob("*.txt"))
    logger.info("Found %s resume(s) in folder: %s\n", len(resume_files), {folder_path})

    results = []
    for file_path in resume_files:
        logger.info("Processing: %s", file_path.name)
        processor = ResumeProcessor(file_path, top_n=top_n)
        logger.info("\n%s", processor.keywords_preview())
        results.append((file_path.name, processor.keywords))

    return results
