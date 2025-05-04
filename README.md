# Personality-Detection
This repository contains a Google Colab notebook demonstrating Personality Detection. It includes code, outputs, and explanations for easy understanding and reproducibility.
This project implements a hybrid NLP pipeline that combines traditional machine learning classifiers with both TF-IDF and BERT-based feature extraction. The goal is to classify textual data accurately, especially when dealing with imbalanced datasets.

Features
Preprocessing using NLTK (tokenization, stopword removal, lemmatization)

Feature extraction via:

TF-IDF Vectorizer

BERT embeddings (via HuggingFace Transformers)

Classification using:

Random Forest

SVM

Class balancing using SMOTE

Evaluation with accuracy and classification reports

Installation
Run the following to install all dependencies:

bash
Copy
Edit
pip install nltk scikit-learn pandas numpy transformers torch imbalanced-learn
Download required NLTK resources:

python
Copy
Edit
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
Usage
Upload your dataset (assumed to be a CSV with text and label columns).

Run the notebook cells sequentially.

Choose between TF-IDF or BERT for feature extraction.

Select a classifier (Random Forest or SVM).

View evaluation metrics.

Project Structure
Text Preprocessing: Tokenization, stopword removal, lemmatization

Feature Extraction: TF-IDF or BERT

Model Training: RandomForestClassifier or SVM

Class Balancing: SMOTE

Evaluation: Accuracy, Precision, Recall, F1-Score
