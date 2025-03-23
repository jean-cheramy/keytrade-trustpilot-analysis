import re
from typing import List, Dict

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')


def preprocess_documents(documents: List[str], language: str = 'english') -> List[str]:
    """
    Preprocess a list of documents by converting to lowercase, removing numbers and punctuation,
    tokenizing the words, and removing stopwords for the specified language.

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


def plot_top_words_by_count(documents: List[str], language: str = 'english', top_n: int = 30,
                            ngram_range: tuple = (1, 1)) -> None:
    """
    Preprocesses the documents, applies CountVectorizer (with n-grams), and plots the top N most frequent n-grams.

    Args:
    documents (List[str]): List of documents to process.
    language (str): Language for stopwords. Default is 'english'.
    top_n (int): Number of top words to plot. Default is 30.
    ngram_range (tuple): Range of n-grams to use. Default is (1, 1) for unigrams.
    """
    preprocessed_docs = preprocess_documents(documents, language)

    # Using CountVectorizer with ngram_range parameter to include n-grams
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    count_matrix = vectorizer.fit_transform(preprocessed_docs)
    feature_names = vectorizer.get_feature_names_out()
    word_counts = np.sum(count_matrix.toarray(), axis=0)
    word_count_df = pd.DataFrame({'word': feature_names, 'count': word_counts})

    # Sort in descending order to put highest counts at the top
    top_words = word_count_df.sort_values('count', ascending=False).head(top_n)

    # Plotting
    plt.figure(figsize=(14, 10))
    colors = plt.cm.viridis(np.linspace(0, 1, top_n))  # Use a fancy color map

    # Create a horizontal bar plot with highest counts at the top
    plt.barh(top_words['word'][::-1], top_words['count'][::-1], color=colors)
    plt.xlabel('Word Count', fontsize=16, labelpad=10)
    plt.ylabel('Words', fontsize=16, labelpad=10)
    plt.title(f'Top {top_n} Most Frequent N-grams Across All Documents', fontsize=18, pad=20)
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)  # Add gridlines for better readability
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    plt.show()


def count_words(documents: List[str], ngram_range: tuple = (1, 1)) -> Dict[str, int]:
    """
    Counts word frequencies and n-gram frequencies in the given documents using CountVectorizer.

    Args:
    documents (List[str]): List of documents to count.
    ngram_range (tuple): Range of n-grams to count. Default is (1, 1) for unigrams.

    Returns:
    Dict[str, int]: A dictionary with the n-grams/words as keys and their frequency as values.
    """
    # Initialize CountVectorizer with the given ngram_range
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    count_matrix = vectorizer.fit_transform(documents)

    # Count n-gram occurrences
    feature_names = vectorizer.get_feature_names_out()
    count_vector = count_matrix.toarray()

    # Aggregate counts by n-gram
    ngram_counts = Counter({feature_names[i]: np.sum(count_vector[:, i]) for i in range(len(feature_names))})

    return ngram_counts


def analyze_language_group(df: pd.DataFrame, languages_dict: Dict[str, str], ngram_range: tuple = (1, 1)) -> None:
    """
    Analyzes and plots the top words by frequency (with n-grams) for each language in the DataFrame.

    Args:
    df (pd.DataFrame): DataFrame containing the review data with 'language' and 'text' columns.
    languages_dict (Dict[str, str]): Dictionary mapping language codes to language names (e.g., "en": "english").
    ngram_range (tuple): Range of n-grams to use. Default is (1, 1) for unigrams. Example: (1, 2) for unigrams and bigrams.
    """
    for language, group in df.groupby('language'):
        print(f"Processing language: {language}")
        # Plotting word frequency graph
        plot_top_words_by_count(group['text'], language=languages_dict.get(language, 'english'),
                                ngram_range=ngram_range)

        # Count and display the word/n-gram frequencies
        print(f"\nWord/N-gram Frequency Count for {language}:")
        ngram_counts = count_words(group['text'], ngram_range=ngram_range)
        for ngram, count in ngram_counts.most_common(10):  # Display top 10 most common n-grams/words
            print(f"{ngram}: {count}")


if __name__ == "__main__":
    # Example usage
    df = pd.read_csv("../data/dataset.csv", sep="\t")
    languages_dict = {"fr": "french", "en": "english", "nl": "dutch"}
    ngram_range = (1, 3)  # You can change this to (1, 1) for unigrams, (1, 2) for unigrams and bigrams, etc.
    analyze_language_group(df, languages_dict, ngram_range)
