import json
import time
from typing import List

import pandas as pd
import requests
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, \
    balanced_accuracy_score

from src.utils.plot import plot_confusion_matrix

LABELS = ['Negative', 'Neutral', 'Positive']


def generate_response(prompt: str, model: str = "sentiment_analyser") -> str:
    """
    Generates a sentiment analysis response for the given prompt by making a request to the sentiment analysis API.

    Args:
    prompt (str): The text input for which the sentiment analysis is to be performed.
    model (str): The model name to be used for sentiment analysis (default is "sentiment_analyser").

    Returns:
    str: The predicted sentiment or an error message if the API request fails.
    """
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        response_text = response.text
        data = json.loads(response_text)
        return data["response"].capitalize()
    else:
        return "Error while generating the answer"


def evaluate_model(true_labels: pd.Series, pred_labels: List[str]) -> None:
    """
    Evaluates the performance of the model by calculating accuracy, balanced accuracy, and generating a classification report.

    Args:
    true_labels (pd.Series): The true sentiment labels of the test set.
    pred_labels (List[str]): The predicted sentiment labels.
    """
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

    plot_confusion_matrix(true_labels, pred_labels, "../plots/distillbert_cm.png", "DistillBert Confusion Matrix")


def main(test_file: str) -> None:
    """
    Main function to load test data, predict sentiment using the model, and evaluate performance.

    Args:
    test_file (str): Path to the test dataset.
    """
    start_time = time.time()

    df = pd.read_csv(test_file, sep="\t")
    # pred_labels = [generate_response(record) for record in df["text"]]
    #
    # with open("ollama_answers.json", "w+", encoding="utf-8") as f:
    #     json.dump(pred_labels, f, ensure_ascii=False, indent=4)
    #
    # elapsed_time = time.time() - start_time
    # print(f"Time spent: {elapsed_time} seconds")
    true_labels = df['true_sentiment']

    with open("ollama_answers.json", "r", encoding="utf-8") as f:
        pred_labels = json.load(f)
    evaluate_model(true_labels, pred_labels)


# Run the main function
if __name__ == "__main__":
    main("../data/balanced_test_set.csv")
