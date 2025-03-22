from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data (run this once)
nltk.download('punkt')
nltk.download('stopwords')


def preprocess_documents(documents, language='english'):
    """
    Preprocess a list of documents for TF-IDF.

    Args:
    documents (list): List of strings, each string is a document.
    language (str): Language for stopwords. Default is 'english'.

    Returns:
    list: List of preprocessed documents.
    """
    # Get stopwords for the specified language
    stop_words = set(stopwords.words(language))

    preprocessed_docs = []

    for doc in documents:
        # Convert to lowercase
        doc = doc.lower()

        # Remove numbers and punctuation
        doc = re.sub(r'[^\w\s]', '', doc)
        doc = re.sub(r'\d+', '', doc)

        # Tokenize
        tokens = word_tokenize(doc)

        # Remove stopwords
        tokens = [token for token in tokens if token not in stop_words]

        # Join tokens back into a string
        preprocessed_doc = ' '.join(tokens)

        preprocessed_docs.append(preprocessed_doc)

    return preprocessed_docs


df = pd.read_csv("data/reviews.csv", sep="\t")

languages_dict = {"fr": "french", "en": "english", "nl": "dutch"}
for language, group in df.groupby('language'):
    preprocessed_docs = preprocess_documents(group["text"], languages_dict[language])

    # Create TfidfVectorizer object
    vectorizer = TfidfVectorizer()

    # Fit and transform the documents
    tfidf_matrix = vectorizer.fit_transform(preprocessed_docs)

    # Get feature names (words)
    feature_names = vectorizer.get_feature_names_out()

    word_scores = np.sum(tfidf_matrix.toarray(), axis=0)

    # Create a DataFrame with words and their scores
    word_score_df = pd.DataFrame({'word': feature_names, 'score': word_scores})

    # Sort by score in descending order and select top 30 words
    top_words = word_score_df.sort_values('score', ascending=False).head(30)

    # Create bar plot
    plt.figure(figsize=(12, 8))
    plt.bar(top_words['word'], top_words['score'])
    plt.xticks(rotation=90)
    plt.xlabel('Words')
    plt.ylabel('TF-IDF Score')
    plt.title('Top 30 Most Important Words Across All Documents')
    plt.tight_layout()
    plt.show()