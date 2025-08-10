import re
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

nltk.download('punkt')
nltk.download('stopwords')


def load_data(csv_file: str) -> pd.DataFrame:
    """
    Loads the CSV file into a pandas DataFrame.

    Args:
        csv_file (str): The path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the CSV data.
    """
    return pd.read_csv(csv_file, sep="\t")


def clean_and_tokenize(text: str) -> list[str]:
    """
    Cleans and tokenizes the text by converting it to lowercase,
    removing non-alphanumeric characters, and splitting it into words.

    Args:
        text (str): The input text to be processed.

    Returns:
        list[str]: List of cleaned and tokenized words.
    """
    text = text.lower()
    text = re.sub(r'[^\w\sÀ-ÿ]', '', text)
    return word_tokenize(text)


def word_frequency_per_language(df: pd.DataFrame) -> dict[str, dict]:
    """
    Computes word frequencies per language in the given DataFrame.
    Concatenates the title and text of all reviews in each language and computes word frequencies.

    Args:
        df (pd.DataFrame): The DataFrame containing review data with language and text.

    Returns:
        dict[str, dict]: A dictionary where keys are language codes,
                         and values are dictionaries with word frequencies per language.
    """
    word_frequencies = {}
    languages_dict = {"fr": "french", "en": "english", "nl": "dutch"}
    for language, group in df.groupby('language'):
        all_text = " ".join(group['title'].fillna('') + " " + group['text'].fillna(''))
        tokens = clean_and_tokenize(all_text)
        stop_words = set(stopwords.words(languages_dict[language]))
        filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]

        word_frequencies[language] = dict(Counter(filtered_tokens))
        #tfidf here

    return word_frequencies


def save_word_frequencies(word_frequencies: dict[str, dict], output_file: str) -> None:
    """
    Saves the word frequencies to a CSV file.

    Args:
        word_frequencies (dict[str, dict]): A dictionary of word frequencies per language.
        output_file (str): Path to the output CSV file.
    """
    all_words = []

    for language, words in word_frequencies.items():
        for word, count in words.items():
            all_words.append({'language': language, 'word': word, 'count': count})

    freq_df = pd.DataFrame(all_words)
    freq_df.to_csv(output_file, index=False, sep="\t")


def generate_wordcloud(language: str, word_frequencies: dict[str, dict]) -> None:
    """
    Generates a word cloud for a specific language.

    Args:
        language (str): The language code for which to generate the word cloud.
        word_frequencies (dict[str, dict]): A dictionary of word frequencies per language.
    """
    word_freq = word_frequencies.get(language, {})
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {language}')
    plt.show()


def generate_barplot(language: str, word_frequencies: dict[str, dict]) -> None:
    """
    Generates a bar plot for the top 15 most frequent words of a language.

    Args:
        language (str): The language code for which to generate the bar plot.
        word_frequencies (dict[str, dict]): A dictionary of word frequencies per language.
    """
    word_freq = word_frequencies.get(language, {})
    top_words = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True)[:25])

    plt.figure(figsize=(10, 6))
    plt.barh(top_words.keys(), top_words.values())
    plt.title(f'Top 15 Most Frequent Words in {language}')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Frequency')
    plt.gca().invert_yaxis()
    plt.show()


def main(csv_file: str, output_file: str) -> None:
    """
    Main function to perform word frequency analysis, save results, and generate visualizations.

    Args:
        csv_file (str): Path to the CSV file containing review data.
        output_file (str): Path to the output CSV file to store word frequencies.
    """
    df = load_data(csv_file)
    word_frequencies = word_frequency_per_language(df)
    save_word_frequencies(word_frequencies, output_file)

    # Generate word clouds and bar plots for each language
    for language in word_frequencies.keys():
        generate_wordcloud(language, word_frequencies)
        generate_barplot(language, word_frequencies)


csv_file = 'data/reviews.csv'
output_file = 'word_frequencies_per_language.csv'
main(csv_file, output_file)
