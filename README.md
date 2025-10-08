# Fake News Detection — Text Classification Project

## Project Overview
This project demonstrates a **text classification pipeline** to detect fake news.  
The goal is to automatically classify news articles as **real** or **fake** using classical NLP techniques:  
- Text preprocessing  
- Feature extraction using **TF-IDF**  
- **Logistic Regression** model training  
- Model evaluation using metrics and confusion matrix  

---

## Dataset
- **Dataset:** [Fake and Real News Dataset (Kaggle)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)  
- **Files:** `True.csv` (real news), `Fake.csv` (fake news)  
- **Goal:** Classify news articles as **real (1)** or **fake (0)**  

---

## Kaggle API Setup Instructions
To allow the script to download the dataset automatically, you need a **Kaggle API key**:

1. **Create a Kaggle account** if you don’t have one: [https://www.kaggle.com/](https://www.kaggle.com/)  
2. **Generate API token**:  
   - Go to your profile → **Account** → **API** → **Create New API Token**  
   - This downloads a `kaggle.json` file  
3. **Place `kaggle.json` in the correct location**:
   - Linux/macOS: `~/.kaggle/kaggle.json`  
   - Windows: `C:\Users\<YourUser>\.kaggle\kaggle.json`  
4. **Set permissions (Linux/macOS)**:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
