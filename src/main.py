from scrapper import scrape_reviews
from data_loader import json_to_csv, create_balanced_test_set
from models import BERT, naive_bayes, ollama


def main():
    scrape_reviews()

    json_to_csv("data/reviews.json", "data/not_cleaned_reviews.csv")
    train_set, test_set = create_balanced_test_set("data/not_cleaned_reviews.csv", "rating")
    train_set.to_csv("data/balanced_train_set.csv", index=False, sep="\t")
    test_set.to_csv("data/balanced_test_set.csv", index=False, sep="\t")

    BERT.main("data/balanced_test_set.csv")
    naive_bayes.main("data/balanced_train_set.csv", "data/balanced_test_set.csv")
    ollama.main("data/balanced_test_set.csv")

if __name__ == "__main__":
  import torch
  print(torch.version.cuda)          # Should show 12.0
  print(torch.cuda.is_available())   # Should be True
  print(torch.cuda.get_arch_list())  # Should include 12.0

