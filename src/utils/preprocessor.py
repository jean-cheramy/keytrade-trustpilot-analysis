import re
import unicodedata
from typing import Set

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")


class TextPreprocessor:
    """
    A text preprocessing utility for cleaning, tokenizing, and normalizing text.
    Supports both stemming and lemmatization and removes stopwords based on the specified language.
    """

    def __init__(self, use_stemming: bool = True, language: str = "english") -> None:
        """
        Initializes the TextPreprocessor with the option to use stemming or lemmatization.

        Args:
            use_stemming (bool): If True, applies stemming; otherwise, applies lemmatization.
            language (str): The language for stopword removal.
        """
        self.use_stemming = use_stemming
        self.language = language
        self.stop_words: Set[str] = set(stopwords.words(self.language))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        """
        Cleans the input text by removing punctuation, numbers, and normalizing accented characters.

        Args:
            text (str): The input text to be cleaned.

        Returns:
            str: The cleaned text.
        """
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\d+", "", text)
        return self.remove_accents(text)

    @staticmethod
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

    def preprocess_text(self, text: str) -> str:
        """
        Tokenizes the cleaned text, removes stopwords, and applies either stemming or lemmatization.

        Args:
            text (str): The input text to be preprocessed.

        Returns:
            str: The fully processed text.
        """
        cleaned_text = self.clean_text(text)
        tokens = word_tokenize(cleaned_text)
        tokens = [word for word in tokens if word not in self.stop_words]

        if self.use_stemming:
            tokens = [self.stemmer.stem(word) for word in tokens]
        else:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]

        return " ".join(tokens)


if __name__ == "__main__":
    text = "j'ai bien mangé de la purée et des bananes garçons"

    preprocessor = TextPreprocessor(use_stemming=False, language="french")
    cleaned_text = preprocessor.preprocess_text(text)
    print("Preprocessed text (lemmatized):", cleaned_text)

    preprocessor_stem = TextPreprocessor(use_stemming=True, language="english")
    cleaned_text_stem = preprocessor_stem.preprocess_text(text)
    print("Preprocessed text (stemmed):", cleaned_text_stem)
