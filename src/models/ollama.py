import os
import json
import time
from typing import List

import pandas as pd
import requests
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, \
    balanced_accuracy_score

from cm_plot import plot_confusion_matrix

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
    report = classification_report(true_labels, pred_labels, target_names=LABELS)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print("Precision, Recall, and F1-Score per Class:")
    print("\nClassification Report:")
    print(report)

    plot_confusion_matrix(true_labels, pred_labels, "src/models/results_plots/llama3_cm.png", "Llama3 Confusion Matrix")


def main(test_file: str) -> None:
    """
    Main function to load test data, generate or load sentiment predictions,
    and evaluate model performance.

    Args:
        test_file (str): Path to the test dataset (.tsv file).
    """
    start_time = time.time()

    df = pd.read_csv(test_file, sep="\t")
    true_labels = df['true_sentiment']

    pred_file = "src/models/ollama_answers.json"

    if os.path.exists(pred_file):
        with open(pred_file, "r", encoding="utf-8") as f:
            pred_labels = json.load(f)
        print("Loaded predictions from cache.")
    else:
        print("Predicting sentiment using Llama3")
        pred_labels = [generate_response(record) for record in df["text"]]

        with open(pred_file, "w", encoding="utf-8") as f:
            json.dump(pred_labels, f, ensure_ascii=False, indent=4)

        elapsed_time = time.time() - start_time
        print(f"Predictions generated and saved. Time spent: {elapsed_time:.2f} seconds")

    evaluate_model(true_labels, pred_labels)
