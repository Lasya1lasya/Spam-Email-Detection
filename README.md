# Spam-Email-Detection
Spam Email Detection using CountVectorizer & Multinomial Naive Bayes
This project is a binary text classification model that identifies whether an email is spam or not spam using Natural Language Processing (NLP) techniques and Machine Learning. It uses CountVectorizer for feature extraction and Multinomial Naive Bayes for classification.
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
## Technologies & Tools Used
 - Python 

 - NumPy, Pandas

- scikit-learn 

- Streamlit web app

- Jupyter Notebook 

- NLP Preprocessing

## Key Features
Preprocessing of email text (lowercasing, punctuation removal, etc.)

Feature extraction using CountVectorizer

Spam classification using Multinomial Naive Bayes

Model evaluation using accuracy score

Achieved ~98% accuracy on test data
