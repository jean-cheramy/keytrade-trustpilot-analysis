import json
import requests
import time
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, balanced_accuracy_score
from src.utils.plot import plot_confusion_matrix

LABELS = ['Negative', 'Neutral', 'Positive']

def generate_response(prompt, model="sentiment_analyser"):
    url = "http://localhost:11434/api/generate"
    headers = {"content-Type": "application/json"}
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

start_time = time.time()
df = pd.read_csv("../data/balanced_test_set.csv", sep="\t")

pred_labels = [generate_response(record) for record in df["text"]]
with open("ollama_answers.json", "w+", encoding="utf-8") as f:
    json.dump(pred_labels, f, ensure_ascii=False, indent=4)

elapsed_time = time.time() - start_time
print(f"Time spent: {elapsed_time} seconds")
# Compute classification metrics

true_labels = df['true_sentiment']

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
