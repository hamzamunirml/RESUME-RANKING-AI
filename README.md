# 🤖 AI Resume Screening & Candidate Ranking System

An NLP/ML-powered system that automatically screens resumes against job
descriptions and ranks candidates by relevance — built with Streamlit,
scikit-learn, and NLTK.

> Paste or upload resumes and a job description, and get instant,
> ranked similarity scores backed by TF-IDF vectorization and cosine
> similarity — plus a breakdown of matched and missing skills.

---

## 🎯 Features

- **Resume Parsing** — extracts text from PDF resumes (`pdfplumber`, with
  `PyPDF2` fallback)
- **Text Preprocessing** — cleaning, tokenization, stopword removal, and
  lemmatization (NLTK)
- **TF-IDF Vectorization** — using a **pretrained vectorizer** trained on
  a reference corpus for stable, realistic similarity scores
- **Cosine Similarity Scoring** — resume ↔ job description matching
- **Candidate Ranking** — sorted results with rank, score, and skill match
- **Skills Detection** — automatic extraction of matched/missing
  technical skills
- **Interactive Web UI** — upload files, paste text, or try sample data
- **Visualizations** — ranking bar charts and similarity heatmaps
- **CSV Export** — for both the web app and the CLI batch runner

---

## 🖥️ Live Demo

👉 https://resume-ranking-ai-2njmwtfr9wzr82sjnrn8zj.streamlit.app/

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Web App | Streamlit |
| NLP | NLTK (tokenization, stopwords, lemmatization) |
| ML | scikit-learn (TF-IDF, cosine similarity) |
| PDF Parsing | pdfplumber, PyPDF2 |
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Testing | pytest, pytest-cov |

---

## 📁 Project Structure

```
Resume-Ranking-AI/
├── data/                   # resume PDFs & job description TXTs
├── models/                 # pretrained TF-IDF vectorizer (tfidf_vectorizer.pkl)
├── notebooks/              # EDA and modeling notebooks
├── scripts/
│   └── build_vectorizer.py # trains and saves the TF-IDF vectorizer
├── src/                    # core pipeline modules
│   ├── parser.py           # PDF text extraction
│   ├── preprocess.py       # text cleaning & normalization
│   ├── similarity.py       # TF-IDF + cosine similarity
│   ├── ranking.py          # candidate ranking logic
│   └── utils.py            # shared helpers
├── tests/                  # pytest test suite
├── streamlit_app.py        # web app entry point
├── main.py                 # CLI batch-processing entry point
└── requirements.txt
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/Resume-Ranking-AI.git
cd Resume-Ranking-AI
```

### 2. Create a virtual environment & install dependencies
```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

### 3. Build the TF-IDF vectorizer (one-time step)
```bash
cd scripts
python build_vectorizer.py
cd ..
```
This creates `models/tfidf_vectorizer.pkl`, which the app loads at
runtime. Re-run this whenever you add more sample resumes/job
descriptions to `data/`.

### 4. Run the web app
```bash
streamlit run streamlit_app.py
```

### 5. (Optional) Run the CLI batch pipeline
Place resume PDFs in `data/resumes/` and job description `.txt` files in
`data/job_descriptions/`, then:
```bash
python main.py
```
Ranked results are exported to `output/ranked_candidates.csv`.

---

## 🧠 How It Works

1. **Extract** — resume PDFs are parsed into raw text.
2. **Clean** — text is lowercased, stripped of special characters,
   tokenized, stopword-filtered, and lemmatized.
3. **Vectorize** — resumes and job descriptions are converted into TF-IDF
   vectors using a vectorizer **pretrained on a reference corpus**
   (not fit fresh on each comparison — this keeps scores stable and
   realistic even when comparing a single resume to a single JD).
4. **Score** — cosine similarity between resume and JD vectors produces a
   match percentage.
5. **Rank** — candidates are sorted by score, with detected skills shown
   alongside.

---

## 🧪 Testing

```bash
python run_tests.py
# or
pytest tests/ --cov=src --cov-report=html
```

> **Note:** `tests/test_similarity.py` currently targets an older version
> of the similarity API and needs updating to match the pretrained-vectorizer
> approach described above — see `PROJECT_DOCUMENTATION.md` for details.

---

## ☁️ Deployment (Streamlit Cloud)

1. Make sure `models/tfidf_vectorizer.pkl` is committed to the repo.
2. Push to GitHub.
3. On [share.streamlit.io](https://share.streamlit.io): **New app** →
   select this repo → set **Main file path** to `streamlit_app.py` →
   **Deploy**.

---

## 📌 Project Status & Notes

For architecture details, the TF-IDF similarity bug that was found and
fixed, known technical debt, and a full deployment checklist, see
[`PROJECT_DOCUMENTATION.md`](./PROJECT_DOCUMENTATION.md).

---

## 👤 Author

Hamza Munir
BS Artificial Intelligence, KFUEIT
[GitHub](https://github.com/hamzamunirml)

---

## 📄 License

This project is available for educational and portfolio purposes.
