import json
import pandas as pd
import requests
import streamlit as st

LABELS = ['Negative', 'Neutral', 'Positive']


def generate_response(review: str, sentiment: str, language: str, prompt: str, temperature: float) -> str:
    """
    Generates a response based on the selected review, sentiment, language, user prompt, and temperature setting.

    Args:
        review (str): The selected review.
        sentiment (str): The associated sentiment.
        language (str): The detected language.
        prompt (str): The user-provided prompt.
        temperature (float): The creativity setting for text generation.

    Returns:
        str: The generated response.
    """
    full_prompt = f"Review: {review}\nOriginal language: {language}\nSentiment: {sentiment}\nUser Question: {prompt}\nAnswer:"
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "Llama3",
        "prompt": full_prompt,
        "temperature": temperature,
        "stream": False
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        data = response.json()
        return data.get("response", "No response generated.")
    else:
        return "Error while generating the answer"


@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Loads and shuffles the dataset from a CSV file.

    Returns:
        pd.DataFrame: The dataset containing shuffled reviews and sentiments.
    """
    df = pd.read_csv("src/data/balanced_test_set.csv", sep="\t")
    return df.sample(frac=1, random_state=None).reset_index(drop=True)  # Shuffle dataset


df = load_data()

st.image("KEYTRADE_logo-scaled.jpg", width=250)
st.title("Sentiment Analysis Review Generator")

limit = st.slider("Limit reviews displayed:", 1, 20, 10)
random_reviews = df.head(limit)

# Scrollable reviews using expander
with st.expander("Show Reviews", expanded=True):
    for index, row in random_reviews.iterrows():
        st.write(f"**Review {index + 1}:** {row['text']}")
        st.write(f"**Sentiment:** {row['true_sentiment']}")
        st.write(f"**Language:** {row['language']}")
        st.markdown("---")

# Select a review (displaying numbers instead of full text)
review_options = {f"Review {i + 1}": text for i, text in enumerate(random_reviews["text"])}
selected_review_label = st.selectbox("Select a review:", list(review_options.keys()))
selected_review_text = review_options[selected_review_label]

temperature = st.slider("Set response creativity (temperature):", 0.0, 1.0, 0.5, 0.05)
user_prompt = st.text_area("Enter your prompt:")

# Generate and display response
if st.button("Generate Answer"):
    selected_review_row = random_reviews[random_reviews["text"] == selected_review_text]
    sentiment = selected_review_row["true_sentiment"].values[0]
    language = selected_review_row["language"].values[0]

    answer = generate_response(selected_review_text, sentiment, language, user_prompt, temperature)

    st.write("### Generated Answer:")
    st.write(answer)
