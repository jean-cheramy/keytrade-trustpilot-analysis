import time
from typing import Dict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB

from src.utils.plot import plot_confusion_matrix
from src.utils.preprocessor import TextPreprocessor


def load_data(train_file: str, test_file: str) -> (pd.DataFrame, pd.DataFrame):
    """
    Load the training and test datasets from the specified files.

    :param train_file: Path to the training dataset.
    :param test_file: Path to the test dataset.
    :return: Tuple containing the training and test datasets as DataFrames.
    """
    train_data = pd.read_csv(train_file, sep="\t")
    test_data = pd.read_csv(test_file, sep="\t")

    if "language" not in train_data.columns or "language" not in test_data.columns:
        raise ValueError("Both train and test sets must have a 'language' column.")

    return train_data, test_data


def preprocess_text_data(X: pd.Series, language: str) -> pd.Series:
    """
    Preprocess the text data by applying the TextPreprocessor to clean the text.

    :param X: A pandas Series containing the raw text data.
    :param language: The language of the text (used to set the correct stopwords).
    :return: A pandas Series containing the cleaned text data.
    """
    preprocessor = TextPreprocessor(use_stemming=True, language=language)
    return X.apply(preprocessor.preprocess_text)


def train_and_evaluate_model_with_cv(X: pd.Series, y: pd.Series, language: str) -> Dict:
    """
    Perform k-fold cross-validation to evaluate the Naive Bayes model on the text data.

    :param X: Preprocessed text data (features).
    :param y: Labels (true sentiments).
    :param language: The language of the data, used for saving results.
    :return: A dictionary containing the evaluation results (average accuracy and classification report).
    """
    tfidf_vectorizer = TfidfVectorizer()
    X_vectorized = tfidf_vectorizer.fit_transform(X)

    model = MultinomialNB()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cross_val_scores = cross_val_score(model, X_vectorized, y, cv=cv, scoring='accuracy')
    accuracy = cross_val_scores.mean()
    print(f"Cross-validated Accuracy for {language}: {accuracy}")

    model.fit(X_vectorized, y)
    y_pred = model.predict(X_vectorized)
    report = classification_report(y, y_pred, zero_division=0)

    plot_confusion_matrix(
        y, y_pred,
        f"../plots/multinomialNB_cm_{language}.png",
        f"MultinomialNB Confusion Matrix ({language})"
    )

    return {
        "model": model,
        "vectorizer": tfidf_vectorizer,
        "accuracy": accuracy,
        "classification_report": report
    }


def main(train_file: str, test_file: str) -> None:
    """
    Main function to load data, train models for each language, and evaluate them.

    :param train_file: Path to the training dataset.
    :param test_file: Path to the test dataset.
    """
    start_time = time.time()
    train_data, test_data = load_data(train_file, test_file)

    languages = train_data["language"].unique()
    language_dict = {"en": "english", "fr": "french", "nl": "dutch"}

    results = {}

    for language in languages:
        print(f"\nTraining model for language: {language}")

        X_train = train_data[train_data["language"] == language]["text"]
        y_train = train_data[train_data["language"] == language]["true_sentiment"]

        X_train_cleaned = preprocess_text_data(X_train, language_dict.get(language, "english"))
        results[language] = train_and_evaluate_model_with_cv(X_train_cleaned, y_train, language)

    elapsed_time = time.time() - start_time
    print(f"\nTotal time spent: {elapsed_time:.2f} seconds")

    for language, result in results.items():
        print(f"\nResults for {language}:")
        print(f"Accuracy: {result['accuracy']}")
        print(f"Classification Report:\n{result['classification_report']}")


if __name__ == "__main__":
    main("../data/balanced_train_set.csv", "../data/balanced_test_set.csv")
