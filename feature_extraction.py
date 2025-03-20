from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder

def extract_features(df, text_column="Review Text", label_column="Label", method="tfidf", max_features=None):
    # Select Feature Extraction Method
    if method == "tfidf":
        vectorizer = TfidfVectorizer(max_features=max_features)
    elif method == "bow":
        vectorizer = CountVectorizer(max_features=max_features)
    else:
        raise ValueError("Invalid method, select either 'tfidf' or 'bow'")

    # Extract features
    X = vectorizer.fit_transform(df[text_column])

    # Label encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[label_column])

    return X, y, vectorizer, label_encoder