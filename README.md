# Sentence Embeddings Using N-Gram Features

A lightweight Natural Language Processing (NLP) project that generates sentence embeddings using classical **N-gram + TF-IDF** features instead of heavy deep learning models.

This project demonstrates that meaningful sentence similarity can be achieved without GPUs or transformer models like BERT.

---

## 🚀 Features
- Unigram, Bigram and Trigram feature extraction
- TF-IDF vector representation
- Sentence similarity using cosine similarity
- Find most similar sentence from dataset
- Interactive command-line interface
- Lightweight and fast (CPU friendly)

---

## 🧠 What are N-Grams?

An **N-gram** is a sequence of N consecutive words in text.

Example:-

Sentence:-
> Machine learning is powerful

| Type | Output |
|-----|------|
| Unigram | machine, learning, is, powerful |
| Bigram | machine learning, learning is, is powerful |
| Trigram | machine learning is, learning is powerful |

N-grams help capture context better than individual words.

---

## ⚙️ Methodology

1. Text Cleaning (lowercase + remove punctuation)
2. Tokenization
3. N-gram vocabulary generation (1–3 grams)
4. TF-IDF vectorization
5. Cosine similarity comparison

---

## 🛠️ Tech Stack
- Python
- NumPy
- Scikit-learn
- Regular Expressions (re)

## 💡 Contributors
👤 **Your Name - Saurabh**  
📧 **Email:- sauravsingh6462@gmail.com**   
🔗 **LinkedIn: https://www.linkedin.com/in/saurabh1826/**
