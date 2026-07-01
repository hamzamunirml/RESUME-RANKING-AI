# AI Resume Screening & Candidate Ranking System

An intelligent system that automates resume screening using Natural Language Processing (NLP) and Machine Learning techniques. The system parses resumes, compares them with job descriptions using TF-IDF vectorization and cosine similarity, and ranks candidates based on relevance.

## 🎯 Features

- **Resume Parsing**: Extracts text from PDF resumes using PyPDF2 and pdfplumber
- **Text Preprocessing**: Clean, tokenize, remove stopwords, and lemmatize text
- **TF-IDF Vectorization**: Convert text documents to numerical vectors
- **Similarity Scoring**: Calculate cosine similarity between resumes and job descriptions
- **Candidate Ranking**: Rank candidates based on similarity scores
- **Skills Analysis**: Extract and analyze skills from resumes
- **Visualization**: Generate charts and graphs for insights
- **Export Results**: Save rankings as CSV files

## 🛠️ Tech Stack

- **Python 3.8+**
- **Pandas, NumPy** - Data manipulation
- **Scikit-learn** - TF-IDF vectorization, cosine similarity
- **NLTK** - Text preprocessing
- **PyPDF2, pdfplumber** - PDF parsing
- **Matplotlib, Seaborn** - Visualization
- **WordCloud** - Word cloud generation

