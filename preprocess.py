import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove stopwords
    text = " ".join([word for word in text.split() if word not in stop_words])

    # Tokenization and lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in text.split()]

    # Rejoin tokens
    return " ".join(tokens)

def preprocess_dataframe(df, text_column="Review Text"):
    df[text_column] = df[text_column].apply(preprocess_text)
    return df