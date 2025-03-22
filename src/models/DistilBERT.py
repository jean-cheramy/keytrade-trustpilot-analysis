import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, balanced_accuracy_score
from src.utils.plot import plot_confusion_matrix
import time

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

LABELS = ['Negative', 'Neutral', 'Positive']


def predict_sentiment(text: str) -> str:
    """
    Predicts the sentiment of a given text using a multilingual BERT model.
    """
    if not isinstance(text, str) or text.strip() == "":
        return "Unknown"

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=-1)

    sentiment_label = LABELS[torch.argmax(probs).item()]

    return sentiment_label


df = pd.read_csv("../data/balanced_test_set.csv", sep="\t")
start_time = time.time()

tqdm.pandas()
df['sentiment'] = df['text'].progress_apply(predict_sentiment)
elapsed_time = time.time() - start_time
print(f"Time spent: {elapsed_time} seconds")

# Compute classification metrics
true_labels = df['true_sentiment']
pred_labels = df['sentiment']

# Calculate accuracy
accuracy = accuracy_score(true_labels, pred_labels)
balanced_accuracy = balanced_accuracy_score(true_labels, pred_labels)

# Calculate precision, recall, and F1-score for each class
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, labels=LABELS, average=None)

# Display classification report
report = classification_report(true_labels, pred_labels, target_names=LABELS)

# Print out results
print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print("Precision, Recall, and F1-Score per Class:")
for i, label in enumerate(LABELS):
    print(f"{label}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1-Score={f1[i]:.4f}")

print("\nClassification Report:")
print(report)
plot_confusion_matrix(true_labels, pred_labels, "../plots/distillbert_cm.png", "DistillBert Confusion Matrix")
