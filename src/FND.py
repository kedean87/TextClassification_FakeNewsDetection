import os
import zipfile
import pandas as pd
import numpy as np
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import nltk
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

class FND():
    def __init__(self):
        # define dataframes to house data before split
        self.df_real = None
        self.df_fake = None
        self.df = None
        
        # Model Datasets
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # vectorized datasets
        self.X_train_vec = None
        self.X_test_vec = None
        
        # Word Vectorizers
        self.vectorizer_tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
        
        # Classifiers
        self.model_LogReg = LogisticRegression(max_iter=300)
        
        # Model To Use
        self.model = None
        
    def download_dataset(self, use_nltk=False):
        # define kaggle dataset to download
        KAGGLE_DATASET = "clmentbisaillon/fake-and-real-news-dataset"
        DATA_DIR = "data"

        os.makedirs(DATA_DIR, exist_ok=True)

        # Check if files already exist
        if not (os.path.exists(os.path.join(DATA_DIR, "True.csv")) and os.path.exists(os.path.join(DATA_DIR, "Fake.csv"))):
            print("Downloading dataset from Kaggle...")
            os.system(f"kaggle datasets download -d {KAGGLE_DATASET} -p {DATA_DIR} --unzip")
        else:
            print("Dataset already exists. Skipping download.")
        
        self.df_real = pd.read_csv(os.path.join(DATA_DIR, "True.csv"))
        self.df_fake = pd.read_csv(os.path.join(DATA_DIR, "Fake.csv"))
        
        # Add labels: 1=real, 0=fake
        self.df_real["label"] = 1
        self.df_fake["label"] = 0

        # Combine datasets
        self.df = pd.concat([self.df_real, self.df_fake], ignore_index=True)
        
        self.df["combined_text"] = self.df["title"].fillna("") + " " + self.df["text"].fillna("")

        # USE NLTK
        def clean_text_nltk(text):
            tokens = word_tokenize(text)
            
            stop_words = set(stopwords.words("english"))
            
            # Keeps words/numbers only and no stopwords
            tokens = [w for w in tokens if w.isalnum() and w not in stop_words]
            clean_text = " ".join(tokens)
            return clean_text
        
        # use re and punctuation
        def clean_text_re(text):
            text = text.lower()
            text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
            return text
        
        # select which text processing method to use
        cleaner = clean_text_nltk if use_nltk else clean_text_re

        self.df["clean_text"] = self.df["combined_text"].apply(cleaner)
    
    def load_model_datasets(self):
        # Split into train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.df["clean_text"], self.df["label"], test_size=0.2, random_state=42
        )
    
    def vectorize(self, vectorizer):
        self.X_train_vec = vectorizer.fit_transform(self.X_train)
        self.X_test_vec = vectorizer.transform(self.X_test)
    
    def load_model(self, model):
        self.model = model
    
    def train(self):
        self.model.fit(self.X_train_vec, self.y_train)
    
    def predict_and_evaluate(self):
        y_pred = self.model.predict(self.X_test_vec)

        print("Classification Report:\n", classification_report(self.y_test, y_pred, target_names=["Fake", "Real"]))
        print("Confusion Matrix:\n", confusion_matrix(self.y_test, y_pred))
    
    def run(self, use_nltk=False):
        self.download_dataset(use_nltk)
        self.load_model_datasets()
        self.vectorize(self.vectorizer_tfidf)
        self.load_model(self.model_LogReg)
        self.train()
        self.predict_and_evaluate()

if __name__ == "__main__":
    fnd = FND()
    fnd.run(use_nltk=False)
