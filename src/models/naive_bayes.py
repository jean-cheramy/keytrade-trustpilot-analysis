import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from src.utils.plot import plot_confusion_matrix
import time

#todo:
# stemming
# tf-idf
# k-fold
# augment dataset, not enough data and 3 different languages!
start_time = time.time()

train_data = pd.read_csv("../data/balanced_train_set.csv", sep="\t")
test_data = pd.read_csv("../data/balanced_test_set.csv", sep="\t")

X_train = train_data["text"]
y_train = train_data["true_sentiment"]
X_test = test_data["text"]
y_test = test_data["true_sentiment"]


vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X_train_vectorized, y_train)

y_pred = naive_bayes_model.predict(X_test_vectorized)

elapsed_time = time.time() - start_time
print(f"Time spent: {elapsed_time} seconds")
# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
plot_confusion_matrix(y_test, y_pred, "../plots/multinomialNB_cm.png", "MultinomialNB Confusion Matrix")
