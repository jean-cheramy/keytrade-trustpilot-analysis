import re
from typing import List, Dict

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')


def preprocess_documents(documents: List[str], language: str = 'english') -> List[str]:
    """
    Preprocess a list of documents for TF-IDF.

    This function will convert the documents to lowercase, remove numbers and punctuation,
    tokenize the words, and remove stopwords for the specified language.

    Args:
    documents (List[str]): List of strings, each string is a document.
    language (str): Language for stopwords. Default is 'english'.

    Returns:
    List[str]: List of preprocessed documents.
    """
    # Get stopwords for the specified language
    stop_words = set(stopwords.words(language))

    preprocessed_docs = []

    for doc in documents:
        doc = doc.lower()
        doc = re.sub(r'[^\w\s]', '', doc)
        doc = re.sub(r'\d+', '', doc)
        tokens = word_tokenize(doc)
        tokens = [token for token in tokens if token not in stop_words]
        preprocessed_doc = ' '.join(tokens)
        preprocessed_docs.append(preprocessed_doc)

    return preprocessed_docs


def plot_top_words_by_tfidf(documents: List[str], language: str = 'english', top_n: int = 30,
                            ngram_range: tuple = (1, 1)) -> None:
    """
    Preprocesses the documents, applies TF-IDF (with n-grams), and plots the top N most important words by TF-IDF score.

    Args:
    documents (List[str]): List of documents to process.
    language (str): Language for stopwords. Default is 'english'.
    top_n (int): Number of top words to plot. Default is 30.
    ngram_range (tuple): Range of n-grams to use. Default is (1, 1) for unigrams. Example: (1, 2) for unigrams and bigrams.
    """
    preprocessed_docs = preprocess_documents(documents, language)

    # Using TfidfVectorizer with ngram_range parameter to include n-grams
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    tfidf_matrix = vectorizer.fit_transform(preprocessed_docs)
    feature_names = vectorizer.get_feature_names_out()
    word_scores = np.sum(tfidf_matrix.toarray(), axis=0)
    word_score_df = pd.DataFrame({'word': feature_names, 'score': word_scores})

    # Sort in descending order to put highest TF-IDF scores at the top
    top_words = word_score_df.sort_values('score', ascending=False).head(top_n)

    # Plotting
    plt.figure(figsize=(14, 10))
    colors = plt.cm.viridis(np.linspace(0, 1, top_n))  # Use a fancy color map

    # Create a horizontal bar plot with highest TF-IDF values at the top
    plt.barh(top_words['word'][::-1], top_words['score'][::-1], color=colors)
    plt.xlabel('TF-IDF Score', fontsize=16, labelpad=10)
    plt.ylabel('Words', fontsize=16, labelpad=10)
    plt.title(f'Top {top_n} Most Important Words (3-grams) Across All Documents', fontsize=18, pad=20)
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)  # Add gridlines for better readability
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    plt.show()


def analyze_language_group(df: pd.DataFrame, languages_dict: Dict[str, str], ngram_range: tuple = (1, 1)) -> None:
    """
    Analyzes and plots the top words by TF-IDF (with n-grams) for each language in the DataFrame.

    Args:
    df (pd.DataFrame): DataFrame containing the review data with 'language' and 'text' columns.
    languages_dict (Dict[str, str]): Dictionary mapping language codes to language names (e.g., "en": "english").
    ngram_range (tuple): Range of n-grams to use. Default is (1, 1) for unigrams. Example: (1, 2) for unigrams and bigrams.
    """
    for language, group in df.groupby('language'):
        print(f"Processing language: {language}")
        plot_top_words_by_tfidf(group['text'], language=languages_dict.get(language, 'english'),
                                ngram_range=ngram_range)


if __name__ == "__main__":
    # Example usage
    df = pd.read_csv("../data/dataset.csv", sep="\t")
    languages_dict = {"fr": "french", "en": "english", "nl": "dutch"}
    ngram_range = (
    1, 3)  # You can change this to (1, 1) for unigrams only, (1, 3) for unigrams, bigrams, and trigrams, etc.
    analyze_language_group(df, languages_dict, ngram_range)
