import streamlit as st
import json
import requests
import pandas as pd

LABELS = ['Negative', 'Neutral', 'Positive']


def generate_response(review: str, sentiment: str, prompt: str) -> str:
    """
    Generates a response based on the selected review, sentiment, and user prompt.
    """
    full_prompt = f"Review: {review}\nSentiment: {sentiment}\nUser Question: {prompt}\nAnswer:"
    url = "http://localhost:11434/api/generate"
    headers = {"content-Type": "application/json"}
    payload = {
        "model": "Llama3",
        "prompt": full_prompt,
        "stream": False
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        response_text = response.text
        data = json.loads(response_text)
        return data["response"]
    else:
        return "Error while generating the answer"


# Load the dataset (Assumes 'balanced_test_set.csv' exists with 'text' and 'sentiment' columns)
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("src/data/balanced_test_set.csv", sep="\t")
    return df


df = load_data()

# Streamlit UI
st.title("Sentiment Analysis Review Generator")

# Number of reviews to display
limit = st.slider("Limit reviews displayed:", 1, 10, 10)

# Display reviews with sentiment
st.write("### Reviews")
for index, row in df.head(limit).iterrows():
    st.write(f"**Review {index + 1}:** {row['text']}")
    st.write(f"**Sentiment:** {row['true_sentiment']}")
    st.markdown("---")

# Select one review to add to the context
selected_review = st.selectbox(
    "Select a review to use in the prompt:",
    df["text"].head(limit).tolist()
)

# User input prompt
user_prompt = st.text_area("Enter your prompt:")

# Generate and display response
if st.button("Generate Answer"):
    sentiment = df[df["text"] == selected_review]["true_sentiment"].values[0]
    answer = generate_response(selected_review, sentiment, user_prompt)
    st.write("### Generated Answer:")
    st.write(answer)
