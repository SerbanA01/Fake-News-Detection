import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import pickle

# Download required NLTK data
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("punkt_tab")


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop_words]
    # Join tokens back into string
    return " ".join(tokens)


# Load the datasets
print("Loading datasets...")
true_df = pd.read_csv("dataset/True.csv")
fake_df = pd.read_csv("dataset/Fake.csv")

# Add labels
true_df["label"] = 1  # 1 for true news
fake_df["label"] = 0  # 0 for fake news

# Combine the datasets
df = pd.concat([true_df, fake_df], ignore_index=True)

# Preprocess the text
print("Preprocessing text...")
df["processed_text"] = df["text"].apply(preprocess_text)

# Split the data
X = df["processed_text"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorize the text
print("Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Random Forest model
print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_vec, y_train)

# Make predictions
print("Making predictions...")
y_pred = rf_model.predict(X_test_vec)

# Evaluate the model
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame(
    {
        "feature": vectorizer.get_feature_names_out(),
        "importance": rf_model.feature_importances_,
    }
)
feature_importance = feature_importance.sort_values("importance", ascending=False)

print("\nTop 10 most important features:")
print(feature_importance.head(10))

# Save the model and vectorizer using pickle
with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
print(
    "\nModel and vectorizer saved as 'random_forest_model.pkl' and 'tfidf_vectorizer.pkl'."
)
