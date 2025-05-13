# ✅ **Smart Resume Analyzer**

A system that can automatically analyze and extract insights from resumes (CVs), match them to job descriptions, and rank them based on relevance.

---

## 📦 Features Covered

- Text preprocessing
- Tokenization
- Named Entity Recognition (NER)
- Text similarity
- Classification
- Summarization
- Deploying NLP applications

---

## 🧠 Phase 1: NLP Foundations (Beginner)

In this phase, we focus on building the foundation of our resume analyzer.

✅ Core functionality:
- Reads resume files (`.txt` or `.pdf`)
- Cleans and tokenizes text
- Extracts important words (naive keyword extractor)
- Identifies candidate names using simple heuristics
- Matches relevant skills from a known skill set

---

## 📈 Roadmap

### 🔖 Step 1: Set Up the Environment

```bash
poetry add nltk spacy pandas pdfplumber
````

Download required NLP models and corpora:

```bash
python -m nltk.downloader all -d ./nltk_data
python -m spacy download en_core_web_sm
```

### 📌 What's Being Installed?

* **NLTK corpora (`nltk.downloader all`)** downloads all datasets, models, and corpora provided by NLTK (Natural Language Toolkit) includes:

  * Tokenizers (e.g., `punkt`)
  * Corpora (e.g., `gutenberg`, `wordnet`)
  * Stopwords, stemmers, lemmatizers
  * Pretrained taggers/parsers

* **spaCy model (`en_core_web_sm`)** download the small English language model and provides:

  * Tokenizer
  * POS Tagger
  * Named Entity Recognizer (NER)
  * Dependency Parser
  * Lightweight word vectors

---

### 🔖 Step 2: Text Preprocessing Pipeline

This step prepares raw resume text for analysis by:

* Lowercasing all letters
* Removing punctuation
* Tokenizing text into words
* Removing stopwords (e.g., "the", "and")
* Lemmatizing (reducing words to their root form)

---

### 🔍 Key Concepts Explained

* **`word_tokenize`**: Splits raw text into individual word tokens.
* **`stopwords.words('english')`**: Filters out high-frequency, low-information words.
* **`WordNetLemmatizer`**: Reduces words to their lemma (e.g., "running" → "run").
* **`str.maketrans('', '', string.punctuation)`**: Removes punctuation using a translation table.

These are essential steps in NLP for transforming messy raw text into clean, structured input for downstream tasks like classification or search.

---