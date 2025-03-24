import time
from typing import Tuple, Set

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB

from src.utils.plot import plot_confusion_matrix
import re
import unicodedata

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer  # Updated to SnowballStemmer
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("stopwords")

# Merged stopwords from English, Dutch, and French
stop_words: Set[str] = set(stopwords.words("english")) | set(stopwords.words("dutch")) | set(stopwords.words("french"))


def clean_text(text: str) -> str:
    """
    Cleans the input text by removing punctuation, numbers, and normalizing accented characters.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove digits
    return remove_accents(text)


def remove_accents(input_str: str) -> str:
    """
    Removes accents from characters in a string.

    Args:
        input_str (str): The input string.

    Returns:
        str: The string without accented characters.
    """
    nfkd_form = unicodedata.normalize("NFKD", input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


def preprocess_text(text: str) -> str:
    """
    Tokenizes the cleaned text, removes stopwords, and applies stemming.

    Args:
        text (str): The input text to be preprocessed.

    Returns:
        str: The fully processed text.
    """
    cleaned_text = clean_text(text)
    tokens = word_tokenize(cleaned_text)
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords

    # Apply Snowball stemming (for English, Dutch, French, etc.)
    stemmer = SnowballStemmer("english")  # You can also use other languages like "french", "dutch", etc.
    tokens = [stemmer.stem(word) for word in tokens]

    return " ".join(tokens)


def load_data(train_file: str, test_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the training and test datasets from the specified files.

    Args:
    train_file (str): Path to the training dataset.
    test_file (str): Path to the test dataset.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: The training and test data as pandas DataFrames.
    """
    train_data = pd.read_csv(train_file, sep="\t")
    test_data = pd.read_csv(test_file, sep="\t")

    return train_data, test_data


def vectorize_text(X_train: pd.Series, X_test: pd.Series) -> Tuple:
    """
    Vectorize the training and test data using CountVectorizer.

    Args:
    X_train (pd.Series): The training text data.
    X_test (pd.Series): The test text data.

    Returns:
    Tuple: The vectorized training and test data as sparse matrices.
    """
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    return X_train_vectorized, X_test_vectorized


def train_naive_bayes(X_train_vectorized: pd.DataFrame, y_train: pd.Series) -> MultinomialNB:
    """
    Train a Multinomial Naive Bayes model on the vectorized training data.

    Args:
    X_train_vectorized (pd.SparseDataFrame): The vectorized training text data.
    y_train (pd.Series): The labels for the training data.

    Returns:
    MultinomialNB: The trained Naive Bayes model.
    """
    naive_bayes_model = MultinomialNB()
    naive_bayes_model.fit(X_train_vectorized, y_train)

    return naive_bayes_model


def evaluate_model(y_test: pd.Series, y_pred: pd.Series) -> None:
    """
    Evaluate the performance of the model.

    Args:
    y_test (pd.Series): The true labels for the test data.
    y_pred (pd.Series): The predicted labels for the test data.
    """
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred, "../plots/multinomialNB_cm.png", "MultinomialNB Confusion Matrix")


def main(train_file: str, test_file: str) -> None:
    """
    Main function to load data, train the Naive Bayes model, and evaluate it.

    Args:
    train_file (str): Path to the training dataset.
    test_file (str): Path to the test dataset.
    """
    start_time = time.time()

    train_data, test_data = load_data(train_file, test_file)

    X_train = train_data["text"].apply(preprocess_text)
    y_train = train_data["true_sentiment"]
    X_test = test_data["text"].apply(preprocess_text)
    y_test = test_data["true_sentiment"]

    X_train_vectorized, X_test_vectorized = vectorize_text(X_train, X_test)
    naive_bayes_model = train_naive_bayes(X_train_vectorized, y_train)
    y_pred = naive_bayes_model.predict(X_test_vectorized)
    elapsed_time = time.time() - start_time
    print(f"Time spent: {elapsed_time} seconds")
    evaluate_model(y_test, y_pred)


if __name__ == "__main__":
    main("../data/balanced_train_set.csv", "../data/balanced_test_set.csv")
