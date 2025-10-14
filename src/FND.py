import os
import zipfile
import pandas as pd
import numpy as np
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import itertools
import joblib

import nltk
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

import seaborn as sns
import matplotlib.pyplot as plt

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
        # self.vectorizer_tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
        self.vectorizers = {
            "Count": CountVectorizer(max_features=5000, stop_words='english'),
            "TF-IDF": TfidfVectorizer(max_features=5000, stop_words='english'),
            "Hashing": HashingVectorizer(n_features=5000, alternate_sign=False)
            }
        
        # Classifiers
        # self.model_LogReg = LogisticRegression(max_iter=300)
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Naive Bayes": MultinomialNB(),
            "Linear SVC": LinearSVC()
            }
        
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
    
    def predict_and_evaluate(self, plot_confusion_matrix=False):
        y_pred = self.model.predict(self.X_test_vec)

        print("Classification Report:\n", classification_report(self.y_test, y_pred, target_names=["Fake", "Real"]))
        print("Confusion Matrix:\n", confusion_matrix(self.y_test, y_pred))
        
        if plot_confusion_matrix:
            cm = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.show()
        
        acc = accuracy_score(self.y_test, y_pred)
        return acc
    
    def run(self, use_nltk=False, plot_confusion_matrix=False):
        self.download_dataset(use_nltk)
        self.load_model_datasets()
        
        results = []

        # Iterate over all combinations
        for vec_name, model_name in itertools.product(self.vectorizers.keys(), self.models.keys()):
            print("\n", vec_name + ",", model_name, "\n")
            self.vectorize(self.vectorizers[vec_name])
            self.load_model(self.models[model_name])
            self.train()
            
            acc = self.predict_and_evaluate(plot_confusion_matrix)
            results.append((vec_name, model_name, acc))
        
        df_results = pd.DataFrame(results, columns=["Vectorizer", "Model", "Accuracy"])
        print(df_results.sort_values(by="Accuracy", ascending=False).reset_index(drop=True))
        
        best_row = df_results.sort_values(by="Accuracy", ascending=False).iloc[0]
        best_vec_name = best_row["Vectorizer"]
        best_model_name = best_row["Model"]
        best_accuracy = best_row["Accuracy"]

        print(f"\nBest combination: {best_vec_name} + {best_model_name} (Accuracy: {best_accuracy:.4f})")

        # Re-train the best model fully
        self.vectorizer = self.vectorizers[best_vec_name]
        X_train_vec = self.vectorizer.fit_transform(self.X_train)
        joblib.dump(self.vectorizer, "best_vectorizer.pkl")
        
        self.load_model(self.models[best_model_name])
        self.train()
        # Save both model and vectorizer
        joblib.dump(self.model, "best_model.pkl")

if __name__ == "__main__":
    fnd = FND()
    fnd.run(use_nltk=False, plot_confusion_matrix=False)
