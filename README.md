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

## 🧠 Phase 2: Intermediate NLP

In this phase, we move beyond basic preprocessing into deeper linguistic analysis and intelligent matching. We'll learn how to use tools like `spaCy` and `scikit-learn` to extract grammatical and semantic information from resumes, and apply text similarity techniques to compare resumes against job descriptions.

---

### ✅ Core Topics Covered

* Part-of-Speech (POS) Tagging
* Named Entity Recognition (NER)
* Text Vectorization (TF-IDF)
* Cosine Similarity
* Resume-to-Job Matching

---

### 🛠️ Tools

* **spaCy** – for POS tagging and NER
* **scikit-learn** – for TF-IDF and similarity scoring

---

### 🔖 Step 1: spaCy Setup

Install and download the spaCy model:

```bash
poetry add spacy
python -m spacy download en_core_web_sm
```

Then, load it in your code:

```python
import spacy
nlp = spacy.load("en_core_web_sm")
```

Inside your class (e.g., `ResumeProcessor`), you can process resume text:

```python
self.spacy_doc = nlp(self.text)
```

---

### 🔖 Step 2: Add POS Tagging and NER

#### 2.1 Part-of-Speech (POS) Tagging

Returns grammatical roles for each token (noun, verb, adjective, etc.):

```python
def extract_pos_tags(self) -> list[tuple[str, str]]:
    return [(token.text, token.pos_) for token in self.spacy_doc]
```

#### 2.2 Named Entity Recognition (NER)

Extracts named entities such as names, dates, organizations:

```python
def extract_named_entities(self) -> list[tuple[str, str]]:
    return [(ent.text, ent.label_) for ent in self.spacy_doc.ents]
```

#### 🧪 Example Usage

```python
print("Named Entities:")
for ent, label in processor.extract_named_entities():
    print(f"  {ent} → {label}")

print("POS Tags (sample):")
for word, tag in processor.extract_pos_tags()[:10]:
    print(f"  {word} → {tag}")
```

---

### 🔖 Step 3: Text Vectorization with TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) transforms resumes and job descriptions into numeric vectors that reflect the importance of each word in context.

```bash
poetry add scikit-learn
```

```python
from sklearn.feature_extraction.text import TfidfVectorizer

texts = [resume1, resume2, job_description]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)
```

---

### 🔖 Step 4: Resume-to-Job Matching using TF-IDF

Now that both resumes and job descriptions are vectorized, you can compute **cosine similarity** between them to quantify how well each resume matches the job description.

This is the core logic behind resume ranking.

```python
from sklearn.metrics.pairwise import cosine_similarity

job_description = "Looking for a Python developer skilled in machine learning and Docker."
resume_texts = [
    "Experienced Python developer with background in ML and Docker.",
    "Frontend developer with React and Angular experience."
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([job_description, *resume_texts])
similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

for idx, score in enumerate(similarity_scores):
    print(f"Resume {idx+1}: Similarity Score = {score:.2f}")
```

#### ✅ Output Example:

```
Resume 1: Similarity Score = 0.76
Resume 2: Similarity Score = 0.12
```

This method gives a **numerical ranking of resumes** based on relevance to a job.

---

### 🔍 Key Concepts Explained

* **POS Tagging**: Assigns grammatical categories to each word. Helps in filtering or rule-based analysis.
* **NER**: Finds real-world entities like people, companies, and dates.
* **TF-IDF**: Scores words based on their uniqueness and frequency. Converts text into numeric form.
* **Cosine Similarity**: Compares two vectors to see how closely they're aligned. Used to rank resumes.
* **TfidfVectorizer**: Converts raw text into vector form for use in similarity models.
