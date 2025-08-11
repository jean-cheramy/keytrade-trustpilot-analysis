# 🧠 Sentiment Analysis of Trustpilot Reviews of Keytrade Bank

I created this project to showcase my end-to-end data skills and because I was personally curious about Keytrade Bank, a new player in Belgium’s banking scene, as I considered switching my own accounts. Analyzing real Trustpilot reviews helped me explore multilingual sentiment analysis and compare NLP models, combining practical data science with my personal interest.

This project explores the sentiment expressed in over 1,000 multilingual reviews from Trustpilot for Keytrade Bank. It combines web scraping, exploratory data analysis, classical and modern NLP models, and generative AI to compare different approaches to sentiment classification and automated response generation.
A pdf [presentation](Sentiment-Analysis-of-Trustpilot-Reviews-for-Keytrade-Bank.pdf) summarizes the methodology, analysis and insights.

## 🔍 Project Objectives

- Collect and preprocess user reviews from Trustpilot  
- Perform multilingual exploratory analysis  
- Compare sentiment classification models: Naive Bayes, BERT, and LLaMA 3  
- Analyze performance, interpretability, and efficiency trade-offs  
- Explore LLM-based response generation  
- Propose an end-to-end sentiment analysis pipeline suitable for production  

## 📈 Features

- **Web Scraping** using `BeautifulSoup` to extract structured review data  
- **Multilingual EDA** across French, Dutch, and English  
- **Advanced preprocessing** including multilingual stemming and vectorization  
- **Model comparison**:  
  - **Naive Bayes** for lightweight baseline  
  - **BERT (XLM-Roberta)** for contextual multilingual understanding  
  - **LLaMA 3** for advanced generation and classification  
- **LLM-based response generation** pipeline (demo prototype)  


## 🛠️ Tech Stack

- Python  
- BeautifulSoup  
- scikit-learn, pytorch, nltk
- Hugging Face Transformers (BERT)  
- LLaMA 3 (via Ollama or local deployment)  
- Pandas, NumPy, Matplotlib, Seaborn
- Streamlit 


## 📂 Project Structure

```bash
keytrade-trustpilot-analysis/
│
├── data/          # Raw and processed review datasets
├── eda/           # EDA notebooks and scripts
├── src/           # Core scraping, preprocessing, modeling scripts
├── app.py         # Demo UI for sentiment analysis
├── README.md
└── requirements.txt
```
## 🚀 Usage

To run this project locally, follow these steps:

### 1. Install Python Dependencies

First, install the required Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Setting Up Ollama with LLaMA3 🦙

To generate responses in this project, a local API powered by **LLaMA3** via [Ollama](https://ollama.com/) is required. Follow the steps below to install Ollama and run the API:

#### a. Install Ollama

Ollama provides an easy way to run large language models locally.

- Visit the official website: [https://ollama.com/download](https://ollama.com/download)
- Download and install Ollama for your operating system (macOS, Linux, or Windows).

Or use a terminal command if supported (macOS/Linux):

```bash
curl -fsSL https://ollama.com/install.sh | sh
```
Note: Make sure Docker is installed and running, as Ollama relies on it under the hood.

#### b. Pull the LLaMA3 Model

Once Ollama is installed, pull the LLaMA3 model (e.g., llama3) by running:

```bash
ollama pull llama3
```

This will download the model and make it available locally.

#### c. Start the API Server

To make the model accessible to the app, run the following in your terminal:

```bash
ollama run llama3
```

This will start a local HTTP server at:

http://localhost:11434

    ⚠️ The app.py file expects this server to be running and reachable at that URL.

#### d. Configuring the Ollama Model for answer generation

To customize the behavior of the LLaMA3 model in Ollama, you can set parameters such as the temperature and system message in your prompt configuration:

- **Temperature:** Controls creativity vs coherence.  
  Set `temperature` to `0.2` for more focused and coherent responses (lower values make answers more deterministic).
- **Instruction**: Have a look at this [resource](https://www.tutorialspoint.com/prompt_engineering/prompt_engineering_designing_effective_prompts.htm) to design the most effective prompt.


## ✅ Results Summary

| Model       | Accuracy | Time     | Notes                               |
|-------------|----------|----------|-------------------------------------|
| ![Naive Bayes](https://img.shields.io/badge/Naive%20Bayes-%23006400?logoColor=white) | 0.86     | 0.6 sec  | Fast, interpretable baseline        |
| BERT        | 0.82     | 23 sec   | Strong multilingual performance     |
| LLaMA 3     | 0.88     | 48 min   | Best accuracy, slow, inconsistent   |

⚠️ Note on LLaMA 3 via Ollama

While LLaMA 3 achieved the highest accuracy in our tests, its classification output was not always consistent when prompted for discrete labels like Positive, Negative, or Neutral. At times, it returned variations such as “It is positive” or even unrelated tokens like “true” which required manual post-processing to clean the predictions before computing metrics. This inconsistency makes it less reliable for automated batch scoring compared to simpler models like Naive Bayes, which produce deterministic and clean outputs without extra handling.

## 🤖 Automated Response Generation

- Built prototype using **LLaMA 3** to generate review replies
- Added constraints for *politeness*, *tone*, and *relevance*
- Highlighted risks: hallucinations, inconsistency, legal liability
- Suggested **human-in-the-loop** system with LLM-assisted templates


## 🚧 Potential Improvements

- Fine-tune multilingual models (e.g., DistilBERT) per language
- Use topic modeling (e.g., BERTopic) on negative reviews
- Integrate with Trustpilot API for live monitoring
- Expand data sources (internal feedback, support tickets)
- Deploy cloud-native solution using Azure (data ingestion, monitoring, scaling)


## 📊 Insights

- Most reviews were either highly positive or very negative, suggesting a polarized customer experience
- Negative reviews tend to be longer and more detailed
- Multilingual sentiment analysis presents model and data imbalance challenges
- **Naive Bayes** is a strong baseline for fast classification, while **LLMs** require resources and oversight


## 💬 Contact

**Made by Jean Cheramy**  
