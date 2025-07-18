import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
from sklearn.model_selection import train_test_split

# Create a sample dataset if spam_email_dataset.csv is not available
data = {
    'Message': [
        'Win a free iPhone now!', 
        'Meeting at 10 AM tomorrow', 
        'Claim your prize today!', 
        'Project update: please review'
    ],
    'Label': [1, 0, 1, 0]  # 1 for spam, 0 for non-spam
}
df = pd.DataFrame(data)
df.to_csv('spam_email_dataset.csv', index=False)

# Load the dataset
data = pd.read_csv('spam_email_dataset.csv')
X = data['Message']
y = data['Label']

# Create a pipeline
pipeline = make_pipeline(
    TfidfVectorizer(max_features=5000),
    MultinomialNB()
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Save the model
joblib.dump(pipeline, 'spam_classifier_model.pkl')
