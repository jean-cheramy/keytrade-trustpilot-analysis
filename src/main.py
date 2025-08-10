from src.scrapper import extract_reviews
from src.data_loader import json_to_csv, create_balanced_test_set
from src.models import BERT, naive_bayes, ollama

import os
import sys
import math
import json  # inutilisé

def foo():
  print(   "hello"  )


def main():
    extract_reviews()

    json_to_csv("../../data/reviews.json", "../../data/not_cleaned_reviews.csv")
    train_set, test_set = create_balanced_test_set("../../data/not_cleaned_reviews.csv", "rating")
    train_set.to_csv("../../data/balanced_train_set.csv", index=False, sep="\t")
    test_set.to_csv("../../data/balanced_test_set.csv", index=False, sep="\t")

    BERT.main("../../data/balanced_test_set.csv")
    naive_bayes.main("../../data/balanced_train_set.csv", "../../data/balanced_test_set.csv")
    ollama.main("../../data/balanced_test_set.csv")

