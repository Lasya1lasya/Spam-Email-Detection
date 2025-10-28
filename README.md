# Spam Email Detection using Machine Learning & Streamlit
## Overview
This project demonstrates a Spam Email Classification System built using Python, Machine Learning (Naive Bayes), and an interactive Streamlit Web App.
It classifies any given email as Spam (1) or Not Spam (0) based on the message content.
The system uses Natural Language Processing (NLP) techniques to convert text into numerical features and achieves ~99% accuracy on real-world email data.
## Datset
| Column   | Description                               |
| -------- | ----------------------------------------- |
| **text** | Email content (subject and body)          |
| **spam** | Target label — 1 for spam, 0 for non-spam |
## Feature Extraction (Text Vectorization)
1) CountVectorizer() converts text into a matrix of word counts (bag-of-words model).
2) Each word becomes a feature; the count of its occurrences in each email becomes a value.
## Train-Test Split
The dataset is split into:
- 80% for training
- 20% for testing
This ensures the model is validated on unseen data.80% for training
##  Model Training
1) MultinomialNB() is ideal for text classification problems.
2) The model learns which words are more likely to appear in spam vs non-spam emails.
## Model Evaluation
Result: 0.9895 (≈98.95% accuracy)
This means your model correctly classifies ~99 out of 100 emails.
## Model Saving
1) You saved the trained model and vectorizer as .pkl files.
2) These files can be loaded later to classify new incoming emails without retraining.
## Model Deployment
Create a Streamlit UI for live predictions.

## How to Run the App Locally
streamlit run app.py

## Libraries Used
- import pandas as pd
- from sklearn.feature_extraction.text import CountVectorizer
- from sklearn.model_selection import train_test_split
- from sklearn.naive_bayes import MultinomialNB
- from sklearn.metrics import accuracy_score
- import joblib

## Key Features
Preprocessing of email text (lowercasing, punctuation removal, etc.)

Feature extraction using CountVectorizer

Spam classification using Multinomial Naive Bayes

Model evaluation using accuracy score

Achieved ~98% accuracy on test data
