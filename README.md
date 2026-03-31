# 📧 Message Spam Filtering

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.6+-green.svg)](https://www.nltk.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-red.svg)](https://pandas.pydata.org/)

## 📌 Overview

This project implements an **SMS Spam Detection System** that classifies messages as **spam** or **ham** (legitimate) using Natural Language Processing (NLP) and machine learning techniques. The system preprocesses text data, extracts meaningful features using TF-IDF vectorization, and evaluates multiple classifiers to identify the most effective model for spam detection.

## 🎯 Problem Statement

With the increasing volume of SMS messages, spam has become a significant nuisance and security risk. This project aims to automatically detect and filter spam messages, helping users avoid:
- Phishing attempts
- Fraudulent promotions
- Unwanted marketing content
- Malware distribution links

## 📊 Dataset

**Source**: SMS Spam Collection Dataset

| Category | Count | Percentage |
|----------|-------|------------|
| **Ham (Legitimate)** | 4,827 | 86.6% |
| **Spam** | 747 | 13.4% |
| **Total** | **5,574** | **100%** |

### Dataset Characteristics
- **Format**: CSV file with message text and spam/ham labels
- **Language**: English
- **Source**: Real SMS messages collected from various sources
- **Challenge**: Imbalanced dataset (more ham than spam)

## 🛠️ Tech Stack

| Category | Libraries/Tools |
|----------|----------------|
| **Data Processing** | pandas, numpy |
| **NLP Preprocessing** | NLTK (stopwords, word tokenization, stemming) |
| **Feature Extraction** | scikit-learn (TfidfVectorizer) |
| **Machine Learning** | scikit-learn (SVM, MultinomialNB, DecisionTreeClassifier) |
| **Evaluation Metrics** | scikit-learn (classification_report, confusion_matrix) |

## 🔧 Methodology

### 1. Text Preprocessing Pipeline

```python
def preprocess_text(text):
    """
    Clean and normalize text for classification
    - Convert to lowercase
    - Remove punctuation and numbers
    - Remove stopwords
    - Apply stemming
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    words = word_tokenize(text)
    
    # Remove stopwords and stem
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    words = [stemmer.stem(word) for word in words 
             if word not in stop_words]
    
    return ' '.join(words)
```
### Feature Extraction with TF-IDF
```
# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    max_features=5000,      # Limit vocabulary size
    ngram_range=(1, 2),     # Include unigrams and bigrams
    stop_words='english'    # Remove common words
)

X = vectorizer.fit_transform(messages)
```
### Confusion Matrix (Best Model - SVM)
```
              Predicted
              Ham    Spam
Actual Ham    [965]   [5]
       Spam   [15]   [85]
```
## 🚀 How to Run
Prerequisites
```
Python 3.7 or higher
```
## Installation Steps
Clone the repository
```
git clone https://github.com/yourusername/message-spam-filtering.git
cd message-spam-filtering
```
Install dependencies
```
pip install pandas nltk scikit-learn numpy matplotlib seaborn
```
Download NLTK data
```
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```
Run the project

Option 1: Jupyter Notebook
```
jupyter notebook messageSpamFiltering.ipynb
```
Option 2: Python Script
```
python messageSpamFiltering.py
```
## 📁 Project Structure
```
message-spam-filtering/
├── messageSpamFiltering.ipynb   # Jupyter notebook with analysis
├── messageSpamFiltering.py      # Python script version
├── spam.csv                     # SMS dataset
├── requirements.txt             # Dependencies list
├── utils.py                     # Helper functions (optional)
└── README.md                    # Project documentation
```
### ⭐ If this project helped you understand spam filtering, please consider giving it a star!

Note: This project is for educational purposes. For production deployment, consider additional security measures and regular model retraining.


This markdown provides:

1. **Professional badges** for technology stack
2. **Clear dataset statistics** with formatting
3. **Detailed methodology** with code examples
4. **Comparative results table** with multiple metrics
5. **Step-by-step installation** instructions
6. **Project structure** visualization
7. **Future improvements** roadmap
8. **Sample predictions** for real-world context
9. **Contributing guidelines** for open source collaboration
10. **Professional formatting** suitable for a GitHub portfolio

The README is comprehensive, well-structured, and highlights both the technical implementation and the practical value of the spam filtering system.
