# ⚖️ Legal Case Document Classifier (NLP)

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![NLP](https://img.shields.io/badge/Task-NLP%20%7C%20Text%20Classification-orange.svg)]()
[![scikit-learn](https://img.shields.io/badge/Library-scikit--learn-F7931E.svg)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLP-NLTK-3776AB.svg)](https://www.nltk.org/)

An end-to-end Natural Language Processing (NLP) system that automatically reads legal case summaries and categorizes them into their respective legal domains (Civil, Criminal, Family, Commercial, Employment, Intellectual Property) along with calibrated prediction confidence scores.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Project Architecture & Pipeline](#-project-architecture--pipeline)
- [Dataset Summary](#-dataset-summary)
- [Text Preprocessing](#-text-preprocessing)
- [Machine Learning Models & Results](#-machine-learning-models--results)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Sample Predictions](#-sample-predictions)
- [Key Takeaways](#-key-takeaways)

---

## 📖 Overview

Legal firms, courts, and legal-tech platforms handle thousands of case filings daily. Manual tagging and routing of these documents is time-consuming and prone to human inconsistency.

This project implements a complete, leak-free machine learning workflow that:
1. Cleans and normalizes noisy real-world legal text.
2. Extracts word-level and phrase-level semantic features using **TF-IDF (unigrams & bigrams)**.
3. Trains and benchmarks 3 distinct classification algorithms:
   - **Logistic Regression**
   - **Support Vector Classifier (Linear SVM)**
   - **Random Forest Classifier**
4. Provides multi-class probabilities and top-$N$ confidence breakdowns for new unseen cases.

---

## 🔄 Project Architecture & Pipeline

```
Raw CSV Data
    │
    ▼
[Data Cleaning] ──► Drop empty records & deduplicate rows
    │
    ▼
[Exploratory Data Analysis] ──► Category distributions, word counts, jurisdictions, yearly trends
    │
    ▼
[Text Normalization] ──► Lowercase, regex filtering, stopword removal, WordNet lemmatization
    │
    ▼
[Train/Test Split] ──► Stratified 80/20 split (preventing data leakage)
    │
    ▼
[Feature Extraction] ──► TF-IDF Vectorizer (sublinear TF, n-gram range (1,2))
    │
    ▼
[Model Training & Comparison] ──► Logistic Regression vs. SVM vs. Random Forest
    │
    ▼
[Evaluation & Confidence] ──► Precision, Recall, F1-Score, Confusion Matrix, Predict Proba
```

---

## 📊 Dataset Summary

The dataset comprises legal cases spanning multiple domains:
- **`Civil`**: Property disputes, torts, negligence, damage claims.
- **`Criminal`**: Offenses, police investigations, charges, bail petitions.
- **`Family`**: Divorce, child custody, maintenance, domestic matters.
- **`Commercial`**: Breach of contract, business disputes, financial recovery.
- **`Employment`**: Wrongful termination, workplace discrimination, wage disputes.
- **`Intellectual Property (IP)`**: Patent infringement, trademark violations, copyright copying.

---

## 🧹 Text Preprocessing

Each case summary is normalized using an NLP cleaning pipeline:
1. **Lowercasing**: Ensures uniform token representation.
2. **Special Character & Number Filtering**: Strips out noise, non-alphabetic symbols, and HTML artifacts (`re.sub(r'[^a-zA-Z\s]', ' ', text)`).
3. **Stopword Removal**: Filters non-informative English stopwords (via NLTK).
4. **Lemmatization**: Reduces words to their morphological base using NLTK's `WordNetLemmatizer`.

---

## 🏆 Machine Learning Models & Results

The models are evaluated on the test set (20% holdout, stratified by category):

| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: |
| **Support Vector Machine (SVM)** | **~90.5%** | **0.908** | **0.905** | **0.906** |
| **Logistic Regression** | **~89.8%** | **0.901** | **0.898** | **0.899** |
| **Random Forest** | **~87.2%** | **0.875** | **0.872** | **0.873** |

> **Note on Performance:** The dataset reflects real-world conditions with natural ambiguity and intentional label noise. An F1-score of ~0.90 demonstrates strong generalization without overfitting.

---

## 📁 Project Structure

```
Legal-Case-Document-Classifier/
├── .gitignore                   # Ignore rules for python cache, checkpoints, and OS files
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── data/
│   └── legal_case_dataset.csv   # Legal case records
└── notebooks/
    └── legal_case_document_classifier.ipynb  # Main analysis, EDA, training & evaluation notebook
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Aditya4426g/Legal-Case-Document-Classifier.git
cd Legal-Case-Document-Classifier
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Jupyter Notebook
```bash
jupyter notebook notebooks/legal_case_document_classifier.ipynb
```

---

## 💡 Sample Predictions

```python
# Example inference output with confidence scores:
Text: "The petitioner filed a petition for divorce, child custody, and maintenance support."
-> Predicted Category: Family (Confidence: 96.2%)
   Top Probabilities:
     - Family                  96.2%
     - Civil                    2.1%
     - Commercial               1.1%

Text: "The company alleges patent and trademark infringement against a competitor for unauthorized copying."
-> Predicted Category: Intellectual Property (Confidence: 94.8%)
   Top Probabilities:
     - Intellectual Property   94.8%
     - Commercial               3.2%
     - Civil                    1.2%
```

---

## 🎯 Key Takeaways & Best Practices

- **Strict Leakage Prevention**: Train/test split is applied prior to fitting the TF-IDF feature extractor.
- **Calibrated Probabilities**: SVM with probability calibration allows confidence tracking across all target classes.
- **Clean Extensibility**: Ready for extensions like deep learning representations (BERT, RoBERTa, LegalBERT).
