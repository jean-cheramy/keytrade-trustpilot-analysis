import time

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, \
    balanced_accuracy_score
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.utils.plot import plot_confusion_matrix

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

LABELS = ['Negative', 'Neutral', 'Positive']


def predict_sentiment(text: str) -> str:
    """
    Predicts the sentiment of a given text using a multilingual BERT model.

    :param text: The text for sentiment analysis.
    :return: The predicted sentiment label ('Negative', 'Neutral', 'Positive') or 'Unknown' if the input is invalid.
    """
    if not isinstance(text, str) or text.strip() == "":
        return "Unknown"

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)
    sentiment_label = LABELS[torch.argmax(probs).item()]

    return sentiment_label


def evaluate_model(df: pd.DataFrame) -> None:
    """
    Evaluates the model by computing various metrics including accuracy, precision, recall, and F1-score.

    :param df: The DataFrame containing the true labels and predicted sentiments.
    """
    true_labels = df['true_sentiment']
    pred_labels = df['sentiment']

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, pred_labels)
    balanced_accuracy = balanced_accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, labels=LABELS, average=None)
    report = classification_report(true_labels, pred_labels, target_names=LABELS)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print("Precision, Recall, and F1-Score per Class:")
    # for i, label in enumerate(LABELS):
    #     print(f"{label}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1-Score={f1[i]:.4f}")

    print("\nClassification Report:")
    print(report)


def predict_and_evaluate(df: pd.DataFrame) -> None:
    """
    Applies sentiment prediction on the input DataFrame and evaluates the model performance.

    :param df: The DataFrame containing the text data and true sentiment labels.
    """
    start_time = time.time()

    tqdm.pandas()
    df['sentiment'] = df['text'].progress_apply(predict_sentiment)

    elapsed_time = time.time() - start_time
    print(f"Time spent: {elapsed_time} seconds")
    evaluate_model(df)
    plot_confusion_matrix(
        df['true_sentiment'], df['sentiment'],
        "../plots/distillbert_cm.png",
        "DistillBert Confusion Matrix"
    )


def main(test_file: str) -> None:
    """
    Main function to load data, predict sentiments, and evaluate the model.

    :param test_file: Path to the test dataset.
    """
    df = pd.read_csv(test_file, sep="\t")
    predict_and_evaluate(df)


if __name__ == "__main__":
    main("../data/balanced_test_set.csv")
