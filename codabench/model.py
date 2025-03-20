import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load training, validation, and test sets
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("validation.csv")
test_df = pd.read_csv("test.csv")

# Extract features using TF-IDF
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_df["Review Text"])
X_val = vectorizer.transform(val_df["Review Text"])
X_test = vectorizer.transform(test_df["Review Text"])

# Get labels
y_train = train_df["Label"]
y_val = val_df["Label"]

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate on the validation set
val_predictions = model.predict(X_val)
print("\nValidation Set Performance:")
print(classification_report(y_val, val_predictions, zero_division=1))
print(f"Validation Accuracy: {accuracy_score(y_val, val_predictions) * 100:.2f}")

# Generate predictions for test set
test_predictions = model.predict(X_test)

# Save test predictions to a CSV
test_predictions_df = pd.DataFrame(test_predictions, columns=["Predicted Label"])
test_predictions_df.to_csv("test_predictions.csv", index=False)
print("\nTest predictions saved to 'test_predictions.csv'")

# Save  vectorizer and model for future
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(model, "logistic_regression_model.pkl")