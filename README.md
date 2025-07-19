# 📧 Spam Email Classifier – A Machine Learning Project in Python

---

## 📌 Project Overview

This project demonstrates a simple yet effective **spam email classifier** using Python and Scikit-learn. It uses natural language processing with TF-IDF vectorization and a Naive Bayes algorithm to detect whether an email message is spam or not.

---

## 🎯 Purpose

To build a lightweight text classification model that identifies spam messages from legitimate ones using machine learning — suitable for learning basic NLP and model deployment.

---

## 🧠 Features

✅ Detects spam vs non-spam emails  
✅ Uses TF-IDF to vectorize message text  
✅ Built using Scikit-learn’s pipeline (clean and efficient)  
✅ Evaluates performance using accuracy & classification report  
✅ Saves the trained model for reuse with `joblib`  

---

## ⚙️ Technologies Used

- **Python 3.6+**
- `pandas` – for dataset handling  
- `scikit-learn` – for ML pipeline, vectorization, and model  
- `joblib` – for saving the trained model  

---

## 🗂️ Dataset

A sample dataset (`spam_email_dataset.csv`) is automatically created if not already present. It includes labeled messages (spam = 1, non-spam = 0) for training/testing the model.

---

## 🚀 How to Run

### 1. Clone the repository

    git clone https://github.com/yourusername/spam-email-classifier.git
    cd spam-email-classifier

## 2. Run the script
 
    python spam_email_classifier.py

The model will train, evaluate accuracy, print a classification report, and save the trained model to spam_classifier_model.pkl.

---
## 📊 Sample Output
        Accuracy: 1.0

        Classification Report:
                       precision    recall  f1-score   support

                 0       1.00      1.00      1.00         1
                 1       1.00      1.00      1.00         0

         accuracy                             1.00          1
        macro avg        1.00      1.00       1.00          1
      weighted avg       1.00      1.00       1.00          1

---

## 📄 License

MIT License — open for use, learning, and modification.

---
## 👨‍💻 Developed By

Shitanshu 

🔗 GitHub: https://github.com/shitanshu15

  
