# Spam Email Classifier

This project is a Python-based spam email classifier that uses machine learning and natural language processing (NLP) techniques to classify emails as spam or non-spam (ham). The classifier is trained on a dataset of labeled email content, and it predicts whether a given email is spam or not.

## Features

- **Text Preprocessing**: The project processes email content by removing punctuation, stopwords, and converting text to lowercase.
- **Feature Extraction**: The model uses **TF-IDF Vectorizer** to convert the email text into numerical features.
- **Machine Learning Model**: A **Naive Bayes classifier** (MultinomialNB) is used to classify emails into two categories: "spam" and "ham".
- **Model Evaluation**: The model is evaluated using accuracy, precision, recall, and F1-score.

## Requirements

To run this project, you need to have the following libraries installed:
- pandas
- scikit-learn
- nltk

You can install these dependencies using `pip`:
```bash
pip install pandas scikit-learn nltk
