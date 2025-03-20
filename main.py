import pandas as pd
from sklearn.model_selection import train_test_split
from preprocess import preprocess_dataframe
from convert_data import parse_reviews
from feature_extraction import extract_features

# Cnvert txt dataset into csv
df = parse_reviews("./data/annotated_reviews.txt", "./data/joint_dataset.csv")

# Load dataset
df = pd.read_csv("./data/joint_dataset.csv")

# Preprocess DataFrame
df = preprocess_dataframe(df, text_column="Review Text")

# Extract features and encode labels
X, y, vectorizer, label_encoder = extract_features(df, text_column="Review Text", label_column="Label", method="tfidf")

# Save preprocessed data
df.to_csv("./data/preprocessed_dataset.csv", index=False)

# Display preprocessed DataFrame
print(df.head())

# Display the shape of the extracted features
print(f"Shape of extracted features: {X.shape}")
print(f"Label Classes: {label_encoder.classes_}")

# Combine features and labels into  single DataFrame
df_features = pd.DataFrame(X.toarray())
df_features["Label"] = y

# Split data into training (70%), validation (15%), test (15%) sets
train_df, temp_df = train_test_split(df_features, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Test set does NOT include labels
test_df.drop(columns=["Label"], inplace=True)

# Training and validation sets include labels
train_df.to_csv("./data/train_set.csv", index=False)
val_df.to_csv("./data/validation_set.csv", index=False)
test_df.to_csv("./data/test_set.csv", index=False)

# Display shapes of splits
print("Training set shape:", train_df.shape)
print("Validation set shape:", val_df.shape)
print("Test set shape:", test_df.shape)