import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, filename, title="Confusion Matrix"):
    """
    Generate and plot a confusion matrix.

    Parameters:
    - y_true: True class labels
    - y_pred: Predicted class labels
    - labels: List of label names
    - title: Title of the confusion matrix plot
    """
    labels = ["Negative", "Neutral", "Positive"]
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Create a heatmap using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar=False,
                annot_kws={'size': 14})

    # Set plot labels and title
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Labels', fontsize=14)
    plt.ylabel('True Labels', fontsize=14)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.savefig(filename)