import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the training, validation, and test sets
train_df = pd.read_csv("./data/train_set.csv")
val_df = pd.read_csv("./data/validation_set.csv")
test_df = pd.read_csv("./data/test_set.csv")

# Separate features (X) and labels (y)
X_train = train_df.drop(columns=["Label"])
y_train = train_df["Label"]
X_val = val_df.drop(columns=["Label"])
y_val = val_df["Label"]
X_test = test_df  # Test set does not have labels

# Train a Naive Bayes model
print("Training Naive Bayes model...")
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate on the validation set
val_predictions = model.predict(X_val)
print("\nValidation Set Performance:")
print(classification_report(y_val, val_predictions))
print(f"Validation Accuracy: {accuracy_score(y_val, val_predictions) * 100:.2f}%")

# Save the trained model
print("\nSaving the model...")
joblib.dump(model, "./models/naive_bayes_model.pkl")
print("Model saved to './models/naive_bayes_model.pkl'")

# Generate predictions for the test set
test_predictions = model.predict(X_test)

# Save test predictions to a CSV file
test_predictions_df = pd.DataFrame(test_predictions, columns=["Predicted Label"])
test_predictions_df.to_csv("./data/test_predictions.csv", index=False)
print("Test predictions saved to './data/test_predictions.csv'")