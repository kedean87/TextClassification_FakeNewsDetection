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

# -----------------------------
# Step 1: Download dataset
# -----------------------------
KAGGLE_DATASET = "clmentbisaillon/fake-and-real-news-dataset"
DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)

# Check if files already exist
if not (os.path.exists(os.path.join(DATA_DIR, "True.csv")) and os.path.exists(os.path.join(DATA_DIR, "Fake.csv"))):
    print("Downloading dataset from Kaggle...")
    os.system(f"kaggle datasets download -d {KAGGLE_DATASET} -p {DATA_DIR} --unzip")
else:
    print("Dataset already exists. Skipping download.")

# -----------------------------
# Step 2: Load datasets
# -----------------------------
df_real = pd.read_csv(os.path.join(DATA_DIR, "True.csv"))
df_fake = pd.read_csv(os.path.join(DATA_DIR, "Fake.csv"))

# Add labels: 1=real, 0=fake
df_real["label"] = 1
df_fake["label"] = 0

# Combine datasets
df = pd.concat([df_real, df_fake], ignore_index=True)

# -----------------------------
# Step 3: Preprocess text
# -----------------------------
df["text_combined"] = df["title"].fillna("") + " " + df["text"].fillna("")

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return text

df["clean_text"] = df["text_combined"].apply(clean_text)

# -----------------------------
# Step 4: Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=0.2, random_state=42
)

# -----------------------------
# Step 5: TF-IDF + Logistic Regression
# -----------------------------
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=300)
model.fit(X_train_vec, y_train)

# -----------------------------
# Step 6: Evaluate
# -----------------------------
y_pred = model.predict(X_test_vec)

print("Classification Report:\n", classification_report(y_test, y_pred, target_names=["Fake", "Real"]))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
